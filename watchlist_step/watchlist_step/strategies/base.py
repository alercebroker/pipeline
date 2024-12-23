import abc
from abc import ABC


class BaseStrategy(ABC):
    @abc.abstractmethod
    def get_coordinates(self, alerts: dict) -> list[tuple]:
        pass

    @abc.abstractmethod
    def get_new_values(self, matches: list[tuple], alerts: dict) -> list[tuple]:
        pass
