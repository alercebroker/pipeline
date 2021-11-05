import pandas as pd

from .base_correction_strategy import BaseCorrectionStrategy


class ATLASCorrectionStrategy(BaseCorrectionStrategy):
    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        return detections
