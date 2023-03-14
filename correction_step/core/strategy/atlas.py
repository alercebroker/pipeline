import numpy as np
import pandas as pd

from .base import BaseStrategy


class ATLASStrategy(BaseStrategy):
    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        detections["corrected"] = False
        detections["parent_candid"] = np.nan
        return detections
