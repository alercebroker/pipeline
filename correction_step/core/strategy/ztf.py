import warnings

import numpy as np
import pandas as pd

from .base import BaseStrategy


class ZTFStrategy(BaseStrategy):
    EXTRA_FIELDS = ["magnr", "sigmagnr", "distnr"]
    DISTANCE_THRESHOLD = 1.4
    ZERO_MAG = 100.

    @classmethod
    def get_first_corrected(cls, df):
        min_candid = df["candid"].values.argmin()
        first_corr = df["corrected"].iloc[min_candid]
        return first_corr

    @property
    def corrected(self) -> pd.Series:
        return self._extra["distnr"] < self.DISTANCE_THRESHOLD

    @property
    def dubious(self):
        negative = ~self.corrected & (self._generic["isdiffpos"] == -1)
        return (~self.corrected & negative) | (self.first & ~self.corrected) | (~self.first & self.corrected)

    @property
    def first(self):
        full = self._generic.assign(corrected=self.corrected)
        return full.groupby(["aid", "fid"])["corrected"].transform(min, "mjd")

    def correction(self) -> pd.DataFrame:
        columns = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
        corrections = pd.DataFrame(columns=columns, index=self._generic.index, dtype=float)

        aux1 = 10 ** (-.4 * self._extra["magnr"])
        aux2 = 10 ** (-.4 * self._generic["mag"])
        aux3 = np.maximum(aux1 + self._generic["isdiffpos"] * aux2, 0.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            corrections["mag_corr"] = -2.5 * np.log10(aux3)

        aux4 = aux2 ** 2 * self._generic["e_mag"] ** 2 - aux1 ** 2 * self._extra["sigmagnr"] ** 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            corrections["e_mag_corr"] = np.sqrt(aux4) / aux3
            corrections["e_mag_corr_ext"] = aux2 * self._generic["e_mag"] / aux3

        bad = (self._extra["magnr"] < 0) & (self._generic["mag"] < 0)

        corrections[bad | self.corrected] = np.nan

        corrections.replace(np.inf, self.ZERO_MAG)
        return corrections

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
