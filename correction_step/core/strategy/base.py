import abc
import pandas as pd


class BaseStrategy(abc.ABC):
    """The Strategy interface declares operations common to all
    supported versions of some algorithm.

    The Context uses this interface to call the algorithm defined
    by Concrete Strategies.
    """

    @abc.abstractmethod
    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
