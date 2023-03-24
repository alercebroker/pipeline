import pandas as pd

from .base import BaseStrategy


class ATLASStrategy(BaseStrategy):
    @property
    def corrected(self) -> pd.Series:
        return super().corrected

    @property
    def dubious(self) -> pd.Series:
        return super().dubious

    @property
    def stellar(self) -> pd.Series:
        return super().stellar

    def _correct(self) -> pd.DataFrame:
        return super()._correct()
