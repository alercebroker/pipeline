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
    def _get_asteroids_idx(df: pd.DataFrame) -> pd.Index:
        """Current classifier quietly drops this case, they must be reinserted manually"""
        idx = df[df["ssdistnr"] != -999].index
        return idx

    @staticmethod
    def _insert_asteroids(probabilities: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
        """Current classifier quietly drops this case, they must be reinserted manually"""
        asteroids = pd.DataFrame(data=0, columns=probabilities.columns, index=idx)
        return pd.concat((probabilities, asteroids))

    @staticmethod
    def _drop_bad_sn(df: pd.DataFrame, probabilities: pd.DataFrame):
        idx = probabilities[probabilities.idxmax(axis=1) == "SN"].index
        snae = df.loc[idx]
        check = snae["isdiffpos"] == 0  # Negative difference
        check |= (snae["sgscore1"] > 0.5) & (snae["distpsnr1"] < 1)  # Near star

        probabilities.drop(check[check].index, inplace=True)

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
        ).sort_values("jd", ascending=True)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = self._get_asteroids_idx(df)
        results = self.model.execute(df)
        results = self._insert_asteroids(results, idx)
        self._drop_bad_sn(df, results)
        return results
