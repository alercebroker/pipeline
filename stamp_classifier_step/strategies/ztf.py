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
        """Returns the index of stamps that should be classified as asteroids.

        Current stamp classifier just quietly drops these objects, but they should be re-inserted after
        prediction is made for the proper classification of these stamps.

        Args:
            df (pd.DataFrame): Data frame of objects to classified (must be passed before executing prediction)

        Returns:
            pd.Index: Index (AIDs) of objects that should be classified as asteroids
        """
        idx = df[df["ssdistnr"] != -999].index
        return idx

    @staticmethod
    def _insert_asteroids(probabilities: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
        """Adds objects that should be classified as asteroids to the predicted probabilities.

        The AIDs should normally be those extracted by `_get_asteroids_idx`.

        Args:
            probabilities (pd.DataFrame): Predictions from the model
            idx (pd.Index): AIDs of objects that should be classified as asteroids

        Returns:
            pd.DataFrame: Same as `probabilities` but with the additional objects added as asteroids
        """
        asteroids = pd.DataFrame(data=0, columns=probabilities.columns, index=idx)
        asteroids["asteroid"] = 1
        return pd.concat((probabilities, asteroids))

    @staticmethod
    def _drop_bad_sn(df: pd.DataFrame, probabilities: pd.DataFrame):
        """Remove certain objects classified as SNe with the highest probability.

        The criteria for removal are that,

        * the magnitude difference is negative, or,
        * the object is too close to a star (given by the value of `sgscore` and `distpsnr1`).

        **Note:** `probabilities` is modified inplace.

        Args:
            df (pd.DataFrame): Data frame of alerts used as input for predictions
            probabilities (pd.DataFrame): Data frame with predicted probabilities
        """
        idx = probabilities[probabilities.idxmax(axis=1) == "SN"].index
        snae = df.loc[idx]
        check = snae["isdiffpos"] == 0  # Negative difference
        check |= (snae["sgscore1"] > 0.5) & (snae["distpsnr1"] < 1)  # Near star

        probabilities.drop(check[check].index, inplace=True)

    def _to_dataframe(self, messages: List[dict]) -> pd.DataFrame:
        """Generate input data frame for model predictor. The output is sorted by date.

        The returned data frame includes all the columns defined in `FIELDS`, taken directly from the
        corresponding field in the alert, as well as those defined in `EXTRA_FIELDS`, also taken directly
        from the alert, but expected nested within the field `extra_fields`. Will raise an error if any of
        these fields are missing from the alert.

        Additionally, it will include the following columns:

        * `oid`: Corresponds to the alert field `aid` (not used in the classifier, but needed for legacy reasons)
        * `jd`: Calculated from the alert field `mjd`
        * `magpsf`: Corresponds to the alert field `mag`
        * `sigmapsf`: Corresponds to the alert field `e_mag`
        * `cutoutScience`: Corresponds to the alert field `science` within `stamps`
        * `cutoutTemplate`: Corresponds to the alert field `template` within `stamps`
        * `cutoutDifference`: Corresponds to the alert field `difference` within `stamps`

        Args:
            messages (list[dict]): List of messages containing alert data

        Returns:
            pd.DataFrame: Data frame used by the model predictor, indexed by ID
        """
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
        """Call the prediction method of the model.

        Includes manual insertion of asteroid probabilities and removal of wrongly classified SNe.

        Args:
            df (pd.DataFrame): Data generated by `_to_dataframe`

        Returns:
            pd.DataFrame: Class probabilities. Its columns should be the classifier's classes, and indexed by AID
        """
        idx = self._get_asteroids_idx(df)
        results = self.model.execute(df)
        results = self._insert_asteroids(results, idx)
        self._drop_bad_sn(df, results)
        return results
