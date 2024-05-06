from ..core.base import FeatureExtractor, AstroObject
import numpy as np
import pandas as pd
from typing import List


class FoldedKimExtractor(FeatureExtractor):
    def __init__(self, bands: List[str], unit: str):
        self.version = "1.0.0"
        self.bands = bands
        valid_units = ["magnitude", "diff_flux"]
        if unit not in valid_units:
            raise ValueError(f"{unit} is not a valid unit ({valid_units})")
        self.unit = unit

    def preprocess_detections(self, detections: pd.DataFrame) -> pd.DataFrame:
        detections = detections[detections["unit"] == self.unit]
        detections = detections[detections["brightness"].notna()]
        return detections

    def compute_features_single_object(self, astro_object: AstroObject):
        period_feature_name = "Multiband_period"
        if period_feature_name not in astro_object.features["name"].values:
            raise Exception("Folded Kim extractor was not provided with period data")
        period = astro_object.features[
            astro_object.features["name"] == period_feature_name
        ]
        period = period["value"][0]
        detections = astro_object.detections
        detections = self.preprocess_detections(detections)

        features = []
        for band in self.bands:
            band_detections = detections[detections["fid"] == band]
            if len(band_detections) <= 2 or np.isnan(period):
                features.append((f"Psi_CS", np.nan, band))
                features.append((f"Psi_eta", np.nan, band))
            else:
                time = band_detections["mjd"].values
                brightness = band_detections["brightness"].values

                folded_time = np.mod(time, 2 * period) / (2 * period)
                sorted_brightness = brightness[np.argsort(folded_time)]
                sigma = np.std(sorted_brightness)
                if sigma != 0.0:
                    m = np.mean(sorted_brightness)
                    lc_len = len(band_detections)
                    s = np.cumsum(sorted_brightness - m) * 1.0 / (lc_len * sigma)
                    psi_cumsum = np.max(s) - np.min(s)
                    sigma_squared = sigma**2
                    psi_eta = (
                        1.0
                        / ((lc_len - 1) * sigma_squared)
                        * np.sum(
                            np.power(sorted_brightness[1:] - sorted_brightness[:-1], 2)
                        )
                    )
                else:
                    psi_cumsum = psi_eta = np.nan

                features.append((f"Psi_CS", psi_cumsum, band))
                features.append((f"Psi_eta", psi_eta, band))

        features_df = pd.DataFrame(data=features, columns=["name", "value", "fid"])

        sids = detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        features_df["sid"] = sid
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )
