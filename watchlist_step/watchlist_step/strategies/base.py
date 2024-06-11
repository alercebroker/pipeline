import abc
from abc import ABC
from typing import List, Tuple


class BaseStrategy(ABC):
    @abc.abstractmethod
    def get_coordinates(self, alerts: dict) -> List[tuple]:
        pass

    @abc.abstractmethod
    def get_new_values(
        self, matches: List[tuple], alerts: dict
    ) -> Tuple[List[tuple], List[dict]]:
        pass
