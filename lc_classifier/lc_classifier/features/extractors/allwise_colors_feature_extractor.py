from ..core.base import FeatureExtractor, AstroObject
from typing import List
import numpy as np
import pandas as pd
from functools import lru_cache


class AllwiseColorsFeatureExtractor(FeatureExtractor):
    version = "1.0.0"

    def __init__(self, bands: List[str]):
        self.bands = bands
        self.allwise_bands = ["W1", "W2", "W3", "W4"]

    def compute_features_single_object(self, astro_object: AstroObject):
        detections = astro_object.detections
        detections = self.preprocess_detections(detections)

        allwise_band_means = []
        for allwise_band in self.allwise_bands:
            if allwise_band in astro_object.metadata["name"].values:
                all_wise_band_mean = astro_object.metadata[
                    astro_object.metadata["name"] == allwise_band
                ]["value"].values[0]
                allwise_band_means.append(all_wise_band_mean)
            else:
                allwise_band_means.append(np.nan)

        band_means = []
        for band in self.bands:
            band_detections = detections[detections["fid"] == band]
            if len(band_detections) == 0:
                band_means.append(np.nan)
            else:
                band_mean = np.mean(band_detections["brightness"])
                band_means.append(band_mean)

        deltas = []
        for i in range(len(self.allwise_bands) - 1):
            deltas.append(allwise_band_means[i] - allwise_band_means[i + 1])

        for allwise_mean in allwise_band_means:
            for band_mean in band_means:
                deltas.append(band_mean - allwise_mean)

        features = np.stack([self._feature_names(), deltas], axis=-1)
        features_df = pd.DataFrame(data=features, columns=["name", "value"])

        features_df["fid"] = None

        sids = detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        features_df["sid"] = sid
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )

    @lru_cache(1)
    def _feature_names(self):
        names = []
        for i in range(len(self.allwise_bands) - 1):
            names.append(f"{self.allwise_bands[i]}-{self.allwise_bands[i+1]}")

        for allwise_name in self.allwise_bands:
            for band_name in self.bands:
                names.append(f"{band_name}-{allwise_name}")

        return names

    def preprocess_detections(self, detections: pd.DataFrame) -> pd.DataFrame:
        detections = detections[detections["unit"] == "magnitude"]
        detections = detections[detections["brightness"].notna()]
        detections = detections[detections["e_brightness"] < 1.0]

        return detections
