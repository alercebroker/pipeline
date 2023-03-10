from magstats_step.strategies.base_strategy import BaseStrategy
from typing import List


class ZTFMagstatsStrategy(BaseStrategy):
    def compute_magstats(self, detections: List[dict], non_detections: List[dict]):
        pass
