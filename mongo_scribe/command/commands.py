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
        self._check_inputs(collection, data, criteria)
        self.collection = collection
        self.criteria = criteria if criteria else {}
        self.data = data

        try:
            options = options if options else {}
            self.options = Options(**options)
        except TypeError:
            print(
                "Some of the options provided are not supported. Using default values."
            )
            self.options = Options()

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
        probabilities = [
            {
                "classifier_name": self.classifier_name,
                "classifier_version": self.classifier_version,
                "class_name": cls,
                "probability": p,
                "ranking": i + 1,
            }
            for i, (cls, p) in enumerate(self._sort())
        ]
        insert = {"$push": {"probabilities": {"$each": probabilities}}}

        # Insert empty probabilities if AID doesn't exist
        upsert = {"$setOnInsert": {"probabilities": []}}
        ops = [
            UpdateOne(self.criteria, upsert, upsert=self.options.upsert),
            UpdateOne(criteria, insert),
        ]

        if self.options.set_on_insert:
            return ops

        for i, (cls, p) in enumerate(self._sort()):
            filters = {
                "el.classifier_name": self.classifier_name,
                "el.classifier_version": self.classifier_version,
                "el.class_name": cls,
            }
            update = {
                "$set": {
                    "probabilities.$[el].probability": p,
                    "probabilities.$[el].ranking": i + 1,
                }
            }
            ops.append(
                UpdateOne(
                    self.criteria,
                    update,
                    array_filters=[filters],
                )
            )
        return ops


class UpdateFeaturesCommand(UpdateCommand):
    """Update Features for a given object.

    The `data` must contain the features version, and an amount of keys that correspond
    to the names of the features, the values for this keys correspond to the value of the
    feature.

    Example `data`:

    .. code-block::
       {
           "features_version": "v1",
           "features": [
                {"name": "feature1", "value": 1.1, "fid": 0},
                {"name": "feature2", "value": null, "fid": 2},
                ...
           ]
       }

    When using the option `set_on_insert`, the features will be added to the object featurs only
    if there wasnt any feature with the same version and name found.

    Using the `upsert` option will create the object if it doesn't already exist.
    """

    type = ValidCommands.update_features

    def _check_inputs(self, collection, data, criteria):
        super()._check_inputs(collection, data, criteria)
        if "features_version" not in data or "features" not in data:
            raise NoFeatureVersionProvidedException()
        self.features_version = data.pop("features_version")

    def get_operations(self) -> list:
        find_existing_criteria = {
            "features.version": {"$ne": self.features_version},
            **self.criteria,
        }
        features = [
            {
                "version": self.features_version,
                "name": feature["name"],
                "value": feature["value"],
                "fid": feature["fid"],
            }
            for feature in self.data["features"]
        ]
        insert = {"$push": {"features": {"$each": features}}}

        # Insert empty features list if AID doesn't exist
        upsert_operation = {"$setOnInsert": {"features": []}}
        ops = [
            UpdateOne(
                self.criteria, upsert_operation, upsert=self.options.upsert
            ),
            UpdateOne(find_existing_criteria, insert),
        ]

        if self.options.set_on_insert:
            return ops

        for feature in self.data["features"]:
            filters = {
                "el.version": self.features_version,
                "el.name": feature["name"],
                "el.fid": feature["fid"]
            }
            update = {
                "$set": {
                    "features.$[el].value": feature["value"],
                }
            }
            ops.append(
                UpdateOne(
                    self.criteria,
                    update,
                    array_filters=[filters],
                )
            )
        return ops
