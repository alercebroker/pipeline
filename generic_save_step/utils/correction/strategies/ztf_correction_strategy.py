import numpy as np
import pandas as pd

from .base_correction_strategy import BaseCorrectionStrategy
from lc_correction.compute import correction, is_dubious, DISTANCE_THRESHOLD


class ZTFCorrectionStrategy(BaseCorrectionStrategy):

    @classmethod
    def get_first_corrected(cls, df):
        min_candid = df["candid"].values.argmin()
        first_corr = df["corrected"].iloc[min_candid]
        return first_corr

    def do_dubious(self, df: pd.DataFrame):
        # was the first detection corrected?
        min_corr = df.groupby(["oid", "fid"], sort=False).apply(self.get_first_corrected)
        min_corr.name = "first_corrected"
        # join with detections dataframe
        df = df.join(min_corr, on=["oid", "fid"], how="left")
        # get dubious data
        dubious = is_dubious(df.corrected, df.isdiffpos, df.first_corrected)
        return dubious

    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        # Retrieve some metadata for do correction
        fields = detections["extra_fields"].apply(lambda x: [x["distnr"],  x["magnr"], x["sigmagnr"]])
        df = pd.DataFrame(list(fields), columns=["distnr", "magnr", "sigmagnr"])
        df["magpsf"] = detections["mag"]
        df["sigmapsf"] = detections["sigmag"]
        df["isdiffpos"] = detections["isdiffpos"]
        # Is possible correct that detection?
        df["corrected"] = df["distnr"] < DISTANCE_THRESHOLD
        # Apply formula of correction
        corrected = df.apply(lambda x: correction(x.magnr, x.magpsf, x.sigmagnr, x.sigmapsf, x.isdiffpos)
                             if x["corrected"]
                             else (np.nan, np.nan, np.nan),
                             axis=1,
                             result_type="expand")
        corrected.columns = ["magpsf_corr", "sigmapsf_corr", "sigmapsf_corr_ext"]
        corrected["corrected"] = df["corrected"]
        # Create new columns for correction fields: use sequential index to join.
        detections = detections.join(corrected)
        # Apply dubious logic
        detections["dubious"] = self.do_dubious(detections)
        del fields
        del df
        del corrected
        return detections
