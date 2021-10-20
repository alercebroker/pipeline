import pandas as pd

from .base_correction_strategy import BaseCorrectionStrategy
from lc_correction.compute import apply_correction, DISTANCE_THRESHOLD


class ZTFCorrectionStrategy(BaseCorrectionStrategy):

    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        # Retrieve some metadata for do correction
        fields = detections["extra_fields"].apply(lambda x: [x["isdiffpos"], x["distnr"],  x["magnr"], x["sigmagnr"]])
        df = pd.DataFrame(list(fields), columns=["isdiffpos", "distnr", "magnr", "sigmagnr"])
        df["magpsf"] = detections["mag"]
        df["sigmapsf"] = detections["sigmag"]
        # Apply formula of correction
        corrected = df.apply(apply_correction, axis=1, result_type="expand")
        corrected.columns = ["mag_corr", "sigmag_corr", "sigmag_corr_ext"]
        corrected["corrected"] = df["distnr"] < DISTANCE_THRESHOLD
        detections[corrected.columns] = corrected
        detections["extra_fields"] = detections.apply(lambda x: {
            **x["extra_fields"],
            "mag_corr": x["mag_corr"],
            "sigmag_corr": x["sigmag_corr"],
            "sigmag_corr_ext": x["sigmag_corr_ext"],
            "corrected": x["corrected"],
        }, axis=1)
        detections.drop(corrected.columns, axis=1, inplace=True)
        del fields
        del df
        del corrected
        return detections
