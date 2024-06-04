import numpy as np
import pandas as pd

from ..core.base import FeatureExtractor, AstroObject
from ..turbofats import FeatureSpace
from typing import List


class TurboFatsExtractor(FeatureExtractor):
    def __init__(self, bands: List[str], unit: str):
        self.version = "1.1.0"
        self.bands = bands

        valid_units = ["magnitude", "diff_flux"]
        if unit not in valid_units:
            raise ValueError(f"{unit} is not a valid unit ({valid_units})")
        self.unit = unit

        feature_names = [
            "Amplitude",
            "AndersonDarling",
            "Autocor_length",
            "Beyond1Std",
            "Con",
            "Eta_e",
            "Gskew",
            "MaxSlope",
            "Mean",
            "Meanvariance",
            "MedianAbsDev",
            "MedianBRP",
            "PairSlopeTrend",
            "PercentAmplitude",
            "Q31",
            "Rcs",
            "Skew",
            "SmallKurtosis",
            "Std",
            "StetsonK",
            "Pvar",
            "ExcessVar",
            "SF_ML_amplitude",
            "SF_ML_gamma",
            "IAR_phi",
            "LinearTrend",
        ]
        self.feature_space = FeatureSpace(feature_names)

    def get_observations(self, astro_object: AstroObject):
        observations = astro_object.detections
        observations = observations[observations["unit"] == self.unit]
        observations = observations[observations["brightness"].notna()]
        observations = observations.drop_duplicates("mjd")
        return observations

    def compute_features_single_object(self, astro_object: AstroObject):
        observations = self.get_observations(astro_object)

        sids = observations["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        feature_dfs = []
        for band in self.bands:
            band_detections = observations[observations["fid"] == band].copy()
            band_detections.sort_values("mjd", inplace=True)
            band_features = self.feature_space.calculate_features(band_detections)

            band_features_df = pd.DataFrame(
                data=band_features, columns=["name", "value"]
            )
            band_features_df["fid"] = band
            band_features_df["sid"] = sid
            band_features_df["version"] = self.version

            feature_dfs.append(band_features_df)

        turbo_fats_features = pd.concat([f for f in feature_dfs if not f.empty], axis=0)
        if self.unit == "diff_flux":
            turbo_fats_features["name"] += "_flux"

        all_features = [astro_object.features, turbo_fats_features]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )
