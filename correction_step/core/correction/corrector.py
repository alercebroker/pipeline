from ..strategy.base import BaseStrategy
import pandas as pd


class Corrector:
    """
    The Corrector defines the interface of interest to clients.
    """

    def __init__(self, strategy: BaseStrategy) -> None:
        """Usually, the Context accepts a strategy through the constructor,
        but also provides a setter to change it at runtime.
        """
        self.strategy = strategy

    @property
    def strategy(self) -> BaseStrategy:
        """The Context maintains a reference to one of the Strategy objects.
        The Context does not know the concrete class of a strategy.
        It should work with all strategies via the Strategy interface.
        """
        return self.__strategy

    @strategy.setter
    def strategy(self, strategy: BaseStrategy) -> None:
        """Allow replacing a Strategy object at runtime."""
        self.__strategy = strategy

    def compute(self, data: pd.DataFrame):
        """The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        if len(data):
            corrected_data = self.strategy.corrected_frame(data)
            return corrected_data
        return data
