import pandas as pd
from abc import ABC, abstractmethod


class BaseCorrectionStrategy(ABC):
    """The Strategy interface declares operations common to all
    supported versions of some algorithm.

    The Context uses this interface to call the algorithm defined
    by Concrete Strategies.
    """

    @abstractmethod
    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
