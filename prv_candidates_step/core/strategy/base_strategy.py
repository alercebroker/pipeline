import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple

class BasePrvCandidatesStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete Strategies.
    """

    @abstractmethod
    def process_prv_candidates(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass
