import abc
from typing import Literal, Union
from dataclasses import dataclass
from mongo_scribe.command.exceptions import (
    NoDataProvidedException,
    UpdateWithNoCriteriaException,
    NoCollectionProvidedException,
    NoClassifierInfoProvidedException,
    NoAlerceIdentificationProvidedException,
)

CommandTypes = Literal["insert", "update", "update_probabilities"]


@dataclass
class Options:
    """
    Plain class containing possible options
    """

    upsert: bool = False


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
        options = None,
    ):
        if collection is None or collection == "":
            raise NoCollectionProvidedException

        if data is None:
            raise NoDataProvidedException

        self.collection = collection
        self.type = command_type
        self.criteria = criteria
        self.data = data

        try:
            self.options = Options(**options) if options is not None else Options()
        except TypeError:
            print("Some of the options provided are not supported. Using default values.")
            self.options = Options()

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
        options=None,
    ):

        if command_type == "update" and criteria is None:
            raise UpdateWithNoCriteriaException

        super().__init__(collection, command_type, criteria, data, options)

    def get_raw_operation(self):
        return self.criteria, self.data


class UpdateProbabilitiesDbCommand(DbCommand):
    def __init__(
        self,
        collection: str,
        command_type: CommandTypes,
        criteria=None,
        data=None,
        options=None
    ):
        if (
            command_type == "update_probabilities"
            and data["classifier"] is None
        ):
            raise NoClassifierInfoProvidedException

        if "aid" not in criteria:
            raise NoAlerceIdentificationProvidedException

        self.classifier = data.pop("classifier")
        super().__init__(collection, command_type, criteria, data, options)

    def get_raw_operation(self):
        return self.classifier, self.criteria, self.data
