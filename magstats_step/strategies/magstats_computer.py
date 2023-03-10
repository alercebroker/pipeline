from .base_strategy import BaseStrategy
import pandas as pd
from typing import List


class MagstatsComputer:
    def __init__(self, strategy: BaseStrategy):
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseStrategy):
        self._strategy = strategy

    def compute_magtats(
        self, detections: List[dict], non_detections: List[dict]
    ) -> pd.DataFrame:
        return self.strategy.compute_magstats(detections, non_detections)
