from typing import Literal
from .exceptions import NoDataProvidedException, UpdateWithNoCriteriaException

CommandTypes = Literal["insert", "update"]


class DbCommand:
    def __init__(self, type: CommandTypes, criteria=None, data=None):
        if data is None:
            raise NoDataProvidedException

        if type == "update" and criteria is None:
            raise UpdateWithNoCriteriaException

        self.type = type
        self.criteria = criteria
        self.data = data
