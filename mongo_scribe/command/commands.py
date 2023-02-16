import abc
from typing import Literal, Union
from mongo_scribe.command.exceptions import (
    NoDataProvidedException,
    UpdateWithNoCriteriaException,
    NoCollectionProvidedException,
    NoClassifierInfoProvidedException,
)

CommandTypes = Literal["insert", "update", "update_probabilities"]


class DbCommand(abc.ABC):
    """
    Plain class that contains the business rules of a Scribe DB Operation.
    Raises a exception when trying to instantiate an invalid command.
    """

    def __init__(
        self,
        collection: str,
        command_type: CommandTypes,
        criteria=None,
        data=None,
    ):
        if collection is None or collection == "":
            raise NoCollectionProvidedException

        if data is None:
            raise NoDataProvidedException

        self.collection = collection
        self.type = command_type
        self.criteria = criteria
        self.data = data

    @abc.abstractmethod
    def get_raw_operation(self) -> Union[dict, tuple]:
        ...


class InsertDbCommand(DbCommand):
    def get_raw_operation(self):
        return self.data


class UpdateDbCommand(DbCommand):
    def __init__(
        self,
        collection: str,
        command_type: CommandTypes,
        criteria=None,
        data=None,
    ):

        if command_type == "update" and criteria is None:
            raise UpdateWithNoCriteriaException

        super().__init__(collection, command_type, criteria, data)

    def get_raw_operation(self):
        return self.criteria, self.data


class UpdateProbabilitiesDbCommand(DbCommand):
    def __init__(
        self,
        collection: str,
        command_type: CommandTypes,
        criteria=None,
        data=None,
    ):
        if (
            command_type == "update_probabilities"
            and data["classifier"] is None
        ):
            raise NoClassifierInfoProvidedException

        super().__init__(collection, command_type, criteria, data)
        self.classifier = data["classifier"]

    def get_raw_operation(self):
        return self.classifier, self.criteria, self.data
