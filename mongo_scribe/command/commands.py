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
        return [
            UpdateOne(
                self.criteria, {"$set": self.data}, upsert=self.options.upsert
            )
        ]


class InsertProbabilitiesCommand(Command):
    """Adds probabilities to array only if the classifier name is not present for document with given criteria.

    The `data` must have the fields `classifier_name` and `classifier_version` (self-explanatory).

    Additional fields in `data` must be pairs of class names from the classifier and the probability value:

    .. code-block::
       {
           "class1": 0.12,
           "class2": 0.23,
           ...
       }

    When getting the operations, the ranking will be automatically included.
    """

    type = ValidCommands.insert_probabilities

    def _check_inputs(self, collection, data, criteria):
        super()._check_inputs(collection, data, criteria)
        if "classifier_name" not in data or "classifier_version" not in data:
            raise NoClassifierInfoProvidedException()
        self.classifier_name = data.pop("classifier_name")
        self.classifier_version = data.pop("classifier_version")

    def _sort(self, reverse=True):
        return sorted(self.data.items(), key=lambda x: x[1], reverse=reverse)

    def get_operations(self) -> list:
        ops = []

        for i, (cls, p) in enumerate(self._sort()):
            criteria = {
                "probabilities": {
                    "$elemMatch": {
                        "classifier_name": {"$ne": self.classifier_name},
                        "classifier_version": {"$ne": self.classifier_version},
                        "class_name": {"$ne": cls},
                    }
                },
                **self.criteria,
            }
            insert = {
                "$push": {
                    "probabilities": {
                        "classifier_name": self.classifier_name,
                        "classifier_version": self.classifier_version,
                        "class_name": cls,
                        "probability": p,
                        "ranking": i + 1,
                    }
                }
            }
            ops.append(UpdateOne(criteria, insert, upsert=self.options.upsert))
        return ops


class UpdateProbabilitiesCommand(InsertProbabilitiesCommand):
    """Updates probabilities for document with given criteria.

    The `data` must have the fields `classifier_name` and `classifier_version` (self-explanatory).

    Additional fields in `data` must be pairs of class names from the classifier and the probability value:

    .. code-block::
       {
           "class1": 0.12,
           "class2": 0.23,
           ...
       }

    When getting the operations, the ranking will be automatically included.
    """

    type = ValidCommands.update_probabilities

    def get_operations(self) -> list:
        ops = super().get_operations()
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
                    upsert=self.options.upsert,
                )
            )
        return ops
