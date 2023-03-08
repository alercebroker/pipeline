import abc
from typing import Union
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
    """
    Plain class that contains the business rules of a Scribe DB Operation.
    Raises an exception when trying to instantiate an invalid command.
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
    def get_raw_operation(self) -> Union[dict, tuple]:
        pass

    @abc.abstractmethod
    def get_operations(self) -> list:
        pass


class InsertDbCommand(Command):
    type = ValidCommands.insert

    def get_raw_operation(self):
        return self.data

    def get_operations(self) -> list:
        return [InsertOne(self.data)]


class UpdateDbCommand(Command):
    type = ValidCommands.update

    def _check_inputs(self, collection, data, criteria):
        super()._check_inputs(collection, data, criteria)
        if not criteria:
            raise UpdateWithNoCriteriaException()

    def get_raw_operation(self):
        return self.criteria, self.data

    def get_operations(self) -> list:
        return [
            UpdateOne(
                self.criteria, {"$set": self.data}, upsert=self.options.upsert
            )
        ]


class UpdateProbabilitiesDbCommand(Command):
    type = ValidCommands.update_probabilities

    def _check_inputs(self, collection, data, criteria):
        super()._check_inputs(collection, data, criteria)
        if "classifier" not in data:
            raise NoClassifierInfoProvidedException()
        self.classifier_name = data.pop("classifier_name")
        self.classifier_version = data.pop("classifier_version")

    def get_raw_operation(self):
        return self.classifier_name, self.criteria, self.data

    def get_operations(self) -> list:
        def sort():
            return sorted(self.data.items(), key=lambda x: x[1], reverse=True)

        ops = []
        for i, (cls, p) in enumerate(sort()):
            array_filter = {
                "el.classifier_name": self.classifier_name,
                "el.classifier_version": self.classifier_version,
                "el.class_name": cls,
            }
            data = {
                "$set": {
                    "probabilities.$[el].probability": p,
                    "probabilities.$[el].ranking": i + 1,
                }
            }
            ops.append(
                UpdateOne(
                    self.criteria,
                    data,
                    array_filters=[array_filter],
                    upsert=self.options.upsert,
                )
            )
        return ops


class InsertProbabilitiesDbCommand(Command):
    type = ValidCommands.insert_probabilities

    def _check_inputs(self, collection, data, criteria):
        super()._check_inputs(collection, data, criteria)
        if "classifier" not in data:
            raise NoClassifierInfoProvidedException()
        self.classifier_name = data.pop("classifier_name")
        self.classifier_version = data.pop("classifier_version")

    def get_raw_operation(self):
        return self.classifier_name, self.criteria, self.data

    def get_operations(self) -> list:
        def sort():
            return sorted(self.data.items(), key=lambda x: x[1], reverse=True)

        ops = []
        criteria = {
            "probabilities.classifier_name": {"$ne": self.classifier_name},
            **self.criteria,
        }
        for i, (cls, p) in enumerate(sort()):
            data = {
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
            ops.append(UpdateOne(criteria, data, upsert=self.options.upsert))
        return ops
