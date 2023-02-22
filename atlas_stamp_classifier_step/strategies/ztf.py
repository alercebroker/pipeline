import os
import sys
from typing import List

import pandas as pd

from .base import BaseStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), "../../model"))

from deployment import StampClassifier


class ZTFStrategy(BaseStrategy):
    FIELDS = ["candid", "ra", "dec", "isdiffpos"]
    EXTRA_FIELDS = [
        "ndethist",
        "ncovhist",
        "jdstarthist",
        "jdendhist",
        "ssdistnr",
        "sgscore1",
        "distpsnr1",
        "sgscore2",
        "distpsnr2",
        "sgscore3",
        "distpsnr3",
        "fwhm",
        "diffmaglim",
        "classtar",
        "chinr",
        "sharpnr",
    ]

    def __init__(self):
        self.model = StampClassifier()
        super().__init__("ztf_stamp_classifier", "1.0.1")

    @staticmethod
    def _set_asteroid_probability(df: pd.DataFrame, probabilities: pd.DataFrame):
        idx = df[df["ssdistnr"] != -999].index
        probabilities.loc[idx] = 0
        probabilities.loc[idx]["asteroid"] = 1

    @staticmethod
    def _filter_bad_sn(df: pd.DataFrame, probabilities: pd.DataFrame):
        idx = probabilities[probabilities.idxmax(axis=1) == "SN"].index
        selection = df.loc[idx]
        criteria = selection["isdiffpos"] == 0  # Negative difference
        criteria |= (selection["sgscore1"] > 0.5) & (
            selection["distpsnr1"] < 1
        )  # Near star

        probabilities.drop(criteria[criteria].index, inplace=True)

    def _to_dataframe(self, messages: List[dict]) -> pd.DataFrame:
        data, index = [], []
        for msg in messages:
            oid = msg["aid"]
            jd = msg["mjd"] + 2400000.5
            mag, e_mag = msg["mag"], msg["e_mag"]
            science = msg["stamps"]["science"]
            template = msg["stamps"]["template"]
            difference = msg["stamps"]["difference"]
            data.append(
                [oid, science, template, difference, jd, mag, e_mag]
                + [msg[field] for field in self.FIELDS]
                + [msg["extra_fields"][field] for field in self.EXTRA_FIELDS]
            )

            index.append(msg["aid"])

        return pd.DataFrame(
            data=data,
            index=index,
            columns=[
                "oid",
                "cutoutScience",
                "cutoutTemplate",
                "cutoutDifference",
                "jd",
                "magpsf",
                "sigmapsf",
            ]
            + self.FIELDS
            + self.EXTRA_FIELDS,
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        results = self.model.execute(df)
        self._set_asteroid_probability(df, results)
        self._filter_bad_sn(df, results)
        print(results)
        return results
