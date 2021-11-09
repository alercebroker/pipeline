import numpy as np
import pandas as pd

from .base_correction_strategy import BaseCorrectionStrategy


class ATLASCorrectionStrategy(BaseCorrectionStrategy):
    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        detections["corrected"] = False
        detections["parent_candid"] = np.nan
        return detections
