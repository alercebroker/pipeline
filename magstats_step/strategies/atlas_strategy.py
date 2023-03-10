from typing import List
from magstats_step.strategies.base_strategy import BaseStrategy


class ATLASMagstatsStrategy(BaseStrategy):
    def compute_magstats(self, detections: List[dict], non_detections: List[dict]):
        pass
