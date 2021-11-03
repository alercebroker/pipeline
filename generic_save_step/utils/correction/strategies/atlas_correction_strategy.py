import numpy as np
import pandas as pd

from .base_correction_strategy import BaseCorrectionStrategy


class ATLASCorrectionStrategy(BaseCorrectionStrategy):
    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        detections["magpsf_corr"] = np.nan
        detections["sigmapsf_corr"] = np.nan
        detections["sigmapsf_corr_ext"] = np.nan
        detections["corrected"] = np.nan
        detections["dubious"] = np.nan
        detections["parent_candid"] = np.nan
        detections["step_id_corr"] = np.nan
        return detections
