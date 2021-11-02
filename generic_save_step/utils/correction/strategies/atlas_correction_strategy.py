import pandas as pd

from .base_correction_strategy import BaseCorrectionStrategy


class AtlasCorrectionStrategy(BaseCorrectionStrategy):
    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        return detections
