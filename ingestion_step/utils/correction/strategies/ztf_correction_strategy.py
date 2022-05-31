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
        min_corr = df.groupby(["oid", "fid"], sort=False).apply(
            self.get_first_corrected
        )
        min_corr.name = "first_corrected"
        # join with detections dataframe
        df = df.join(min_corr, on=["oid", "fid"], how="left")
        # get dubious data
        dubious = is_dubious(df.corrected, df.isdiffpos, df.first_corrected)
        return dubious

    def do_correction(self, detections: pd.DataFrame) -> pd.DataFrame:
        # Retrieve some metadata for do correction
        fields = detections["extra_fields"].apply(
            lambda x: [x["distnr"], x["magnr"], x["sigmagnr"]]
        )
        # Create an auxiliary dataframe for correction
        df = pd.DataFrame(
            list(fields), columns=["distnr", "magnr", "sigmagnr"]
        )
        # Uses candid like index
        df.index = detections["candid"]
        # Additional columns for correction
        df.loc[:, ["magpsf", "sigmapsf"]] = detections[["mag", "e_mag"]].values
        df.loc[:, "isdiffpos"] = detections["isdiffpos"].values
        # Is possible correct that detection?
        df["corrected"] = df["distnr"] < DISTANCE_THRESHOLD
        # Apply formula of correction: corrected is the dataframe with response
        corrected = df.apply(
            lambda x: correction(
                x.magnr, x.magpsf, x.sigmagnr, x.sigmapsf, x.isdiffpos
            )
            if x["corrected"]
            else (np.nan, np.nan, np.nan),
            axis=1,
            result_type="expand",
        )
        corrected.columns = [
            "magpsf_corr",
            "sigmapsf_corr",
            "sigmapsf_corr_ext",
        ]
        corrected["corrected"] = df["corrected"]
        # Create new columns for correction fields: use candid index to join.
        detections = detections.set_index("candid")
        detections = detections.join(corrected)
        # Reset index and get candid column again
        detections.reset_index(inplace=True)
        # Apply dubious logic
        detections["dubious"] = self.do_dubious(detections)
        # Move correction field to extra_fields
        detections["extra_fields"] = detections.apply(
            lambda x: {
                **x["extra_fields"],
                "magpsf_corr": x["magpsf_corr"],
                "sigmapsf_corr": x["sigmapsf_corr"],
                "sigmapsf_corr_ext": x["sigmapsf_corr_ext"],
                "dubious": x["dubious"],
            },
            axis=1,
        )
        # Remove correction columns of dataframe
        detections.drop(
            columns=[
                "magpsf_corr",
                "sigmapsf_corr",
                "sigmapsf_corr_ext",
                "dubious",
            ],
            inplace=True,
        )
        del fields
        del df
        del corrected
        return detections
