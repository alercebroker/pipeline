import abc
from dataclasses import dataclass

from pymongo.operations import InsertOne, UpdateOne

from .exceptions import *
from .commons import ValidCommands


@dataclass
class Options:
    """
    Plain class containing possible options
    """

    upsert: bool = False
    set_on_insert: bool = False


class Command(abc.ABC):
    """Creates a base command.

    The fields `collection` and `data` are required. The `collection` refers to the collection name in a
    Mongo database and the `data` has forms specified in subclasses.

    The field `criteria` is only relevant for specific subclasses.

    Finally, `options` can be a dictionary with possible additional settings defined in the class `Options`.
    Whether a specific option is used or not, will depend on subclass implementation. If unrecognized options
    or wrong types are provided, the command will use the default options.
    """

    type: str

    def __init__(
        self,
        collection: str,
        data,
        criteria=None,
        options=None,
    ):
        self.criteria = criteria if criteria else {}
        self._check_inputs(collection, data, criteria)
        self.collection = collection
        self.data = self._format_data(data)

        try:
            options = options if options else {}
            self.options = Options(**options)
        except TypeError:
            print(
                "Some of the options provided are not supported. Using default values."
            )
            self.options = Options()

    def _format_data(self, data):
        return data

    def _check_inputs(self, collection, data, criteria):
        if not collection:
            raise NoCollectionProvidedException()
        if not data:
            raise NoDataProvidedException()

    @abc.abstractmethod
    def get_operations(self) -> list:
        pass


class InsertCommand(Command):
    """Directly inserts `data` into the database"""

    type = ValidCommands.insert

    def get_operations(self) -> list:
        return [InsertOne(self.data)]


class UpdateCommand(Command):
    """Updates object in database based on a given criteria.

    Uses MongoDB `$set` operator over the given `data`.
    """

    type = ValidCommands.update

    def __get_forced_photometry_criteria(self, candid):
        return {"_id": candid}

    def __format_forced_photometry(self, data, candid, oid):
        if isinstance(data, list):
            new_data = []
            for d in data:
                data.append({"oid": oid, **d})
        else:
            new_data = {**data, "oid": oid}
        return new_data

    def _format_data(self, data):
        if self.collection == "forced_photometry":
            candid = self.criteria.get("candid")
            oid = self.criteria.get("oid")
            data = self.__format_forced_photometry(data, candid, oid)
            self.criteria = self.__get_forced_photometry_criteria(candid)

        return data

    def _check_inputs(self, collection, data, criteria):
        super()._check_inputs(collection, data, criteria)
        if not criteria:
            raise UpdateWithNoCriteriaException()

    def get_operations(self) -> list:
        op = "$setOnInsert" if self.options.set_on_insert else "$set"
        return [
            UpdateOne(
                self.criteria, {op: self.data}, upsert=self.options.upsert
            )
        ]


""" Note:
Old probability schema:
{
    probabilities: [
        {
            "classifier_name": "name1",
            "classifier_version": "1.0.0",
            "class_name": "AGN",
            "probability": 0.99,
            "ranking": 1, 
        },
        ...
    ]
}

New probability schema:
{
    probabilities: [{
        {
            "classifier_name": "classifier",
            "version": "1.0.0",
            "class_rank_1": "AGN",
            "probability_rank_1": 0.99,
            "values": [{ "class_name": "AGN", "probability": 0.99, "ranking": 1 }, ...]
        },
        ...
    }]
}
"""


class UpdateProbabilitiesCommand(UpdateCommand):
    """Update probabilities for a given object.

    The `data` must have the fields `classifier_name` and `classifier_version`.

    Additional fields in `data` must be pairs of class names from the classifier and the probability value.

    Example `data`:

    .. code-block::
       {
           "classifier_name": "classifier",
           "classifier_version": "1.0.0",
           "class1": 0.12,
           "class2": 0.23,
           ...
       }

    When getting the operations, the ranking will be automatically included.

    When using the option `set_on_insert`, probabilities will be added only if the classifier name cannot be
    found among the existing probabilities.

    Using the `upsert` option will create the object if it doesn't already exist.
    """

    type = ValidCommands.update_probabilities

    def _check_inputs(self, collection, data, criteria):
        super()._check_inputs(collection, data, criteria)
        if "classifier_name" not in data or "classifier_version" not in data:
            raise NoClassifierInfoProvidedException()
        self.classifier_name = data.pop("classifier_name")
        self.classifier_version = data.pop("classifier_version")

    def _sort(self, reverse=True):
        return sorted(self.data.items(), key=lambda x: x[1], reverse=reverse)

    def get_operations(self) -> list:
        criteria = {
            "probabilities.classifier_name": {"$ne": self.classifier_name},
            **self.criteria,
        }

        # from now on we'll be assuming that the first element is the rank one class
        probabilities = [
            {
                "class_name": class_name,
                "probability": p,
                "ranking": i + 1,
            }
            for i, (class_name, p) in enumerate(self._sort())
        ]

        operation = {
            "$push": {
                "probabilities": {
                    "classifier_name": self.classifier_name,
                    "version": self.classifier_version,
                    "class_rank_1": probabilities[0]["class_name"],
                    "probability_rank_1": probabilities[0]["probability"],
                    "values": probabilities,
                }
            }
        }

        # this operation does nothing when there's a classifier
        # with that name previously inserted
        ops = [
            UpdateOne(
                self.criteria,
                {"$setOnInsert": {"probabilities": []}},
                upsert=self.options.upsert,
            ),
            UpdateOne(criteria, operation),
        ]

        filters = {
            "el.classifier_name": self.classifier_name,
        }

        update = {
            "$set": {
                "probabilities.$[el].version": self.classifier_version,
                "probabilities.$[el].class_rank_1": probabilities[0][
                    "class_name"
                ],
                "probabilities.$[el].probability_rank_1": probabilities[0][
                    "probability"
                ],
                "probabilities.$[el].values": probabilities,
            }
        }

        ops.append(UpdateOne(self.criteria, update, array_filters=[filters]))
        return ops


class UpdateFeaturesCommand(UpdateCommand):
    """Update Features for a given object.

    The `data` must contain the features version, features group name and an amount of keys that correspond
    to the names of the features, the values for these keys correspond to the value of the
    feature.

    Example `data`:

    .. code-block::
       {
           "features_version": "v1",
           "features_group": "group",
           "features": [
                {"name": "feature1", "value": 1.1, "fid": 0},
                {"name": "feature2", "value": null, "fid": 2},
                ...
           ]
       }

    When using the option `set_on_insert`, the features will be added to the object features only
    if there wasn't any feature with the same version and name found.

    Using the `upsert` option will create the object if it doesn't already exist.
    """

    type = ValidCommands.update_features

    def _check_inputs(self, collection, data, criteria):
        super()._check_inputs(collection, data, criteria)
        if "features" not in data:
            raise NoFeatureProvidedException()
        if "features_version" not in data:
            raise NoFeatureVersionProvidedException()
        if "features_group" not in data:
            raise NoFeatureGroupProvidedException()
        self.features_version = data.pop("features_version")
        self.features_group = data.pop("features_group")

    def get_operations(self) -> list:
        features = {
            "survey": self.features_group,
            "version": self.features_version,
            "features": self.data["features"],
        }

        criteria = {
            "features.survey": {"$ne": features["survey"]},
            **self.criteria,
        }

        insert = {"$push": {"features": features}}

        ops = [
            UpdateOne(
                self.criteria,
                {"$setOnInsert": {"features": []}},
                upsert=self.options.upsert,
            ),
            UpdateOne(criteria, insert),
        ]

        filters = {"el.survey": features["survey"]}

        update = {
            "$set": {
                "features.$[el].version": features["version"],
                "features.$[el].features": features["features"],
            }
        }

        ops.append(UpdateOne(self.criteria, update, array_filters=[filters]))

        return ops
