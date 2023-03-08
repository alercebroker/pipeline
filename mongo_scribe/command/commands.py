import abc
from typing import Literal, Union
from dataclasses import dataclass


from .exceptions import *
from .commons import ValidCommands


@dataclass
class Options:
    """
    Plain class containing possible options
    """

    upsert: bool = False


class DbCommand(abc.ABC):
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
            print("Some of the options provided are not supported. Using default values.")
            self.options = Options()

    def _check_inputs(self, collection, data, criteria):
        if not collection:
            raise NoCollectionProvidedException()
        if not data:
            raise NoDataProvidedException()

    @abc.abstractmethod
    def get_raw_operation(self) -> Union[dict, tuple]:
        pass


class InsertDbCommand(DbCommand):
    type = ValidCommands.insert

    def get_raw_operation(self):
        return self.data


class UpdateDbCommand(DbCommand):
    type = ValidCommands.update

    def _check_inputs(self, collection, data, criteria):
        super()._check_inputs(collection, data, criteria)
        if not criteria:
            raise UpdateWithNoCriteriaException()

    def get_raw_operation(self):
        return self.criteria, self.data


class UpdateProbabilitiesDbCommand(DbCommand):
    type = ValidCommands.update_probabilities

    def _check_inputs(self, collection, data, criteria):
        super()._check_inputs(collection, data, criteria)
        if "classifier" not in data:
            raise NoClassifierInfoProvidedException()
        self.classifier = data.pop("classifier")
        if "aid" not in criteria:
            raise NoAlerceIdentificationProvidedException()

    def get_raw_operation(self):
        return self.classifier, self.criteria, self.data
