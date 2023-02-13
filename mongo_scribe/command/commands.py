from typing import Literal
from mongo_scribe.command.exceptions import (
    NoDataProvidedException,
    UpdateWithNoCriteriaException,
    NoCollectionProvidedException,
)

CommandTypes = Literal["insert", "update"]


class DbCommand:
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

        if command_type == "update" and criteria is None:
            raise UpdateWithNoCriteriaException

        self.collection = collection
        self.type = command_type
        self.criteria = criteria
        self.data = data
