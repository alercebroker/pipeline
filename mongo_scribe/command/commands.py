from typing import Literal
from .exceptions import NoDataProvidedException, UpdateWithNoCriteriaException, NoCollectionProvidedException

CommandTypes = Literal["insert", "update"]


class DbCommand:
    def __init__(self, collection: str, type: CommandTypes, criteria=None, data=None):
        if collection is None or collection == "":
            raise NoCollectionProvidedException
        
        if data is None:
            raise NoDataProvidedException

        if type == "update" and criteria is None:
            raise UpdateWithNoCriteriaException

        self.collection = collection
        self.type = type
        self.criteria = criteria
        self.data = data

