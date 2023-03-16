import pandas as pd

from .base import BaseStrategy


class ATLASStrategy(BaseStrategy):
    @property
    def near_source(self) -> pd.Series:
        return super().near_source

    @property
    def dubious(self) -> pd.Series:
        return super().dubious

    def _correct(self) -> pd.DataFrame:
        return super()._correct()
