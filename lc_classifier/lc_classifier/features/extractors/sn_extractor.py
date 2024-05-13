import numpy as np
import pandas as pd

from ..core.base import FeatureExtractor, AstroObject
from typing import List, Tuple, Dict


class SNExtractor(FeatureExtractor):
    def __init__(self, bands: List[str], unit: str, use_forced_photo: bool):
        self.version = "1.0.0"
        self.bands = bands
        valid_units = ["diff_flux"]
        if unit not in valid_units:
            raise ValueError(f"{unit} is not a valid unit ({valid_units})")
        self.unit = unit
        self.use_forced_photo = use_forced_photo

        self.detections_min_len = 1

    def compute_features_single_object(self, astro_object: AstroObject):
        detections = astro_object.detections
        detections = detections[detections["unit"] == self.unit]
        detections = detections.sort_values("mjd")
        forced_photometry = astro_object.forced_photometry
        first_detection = detections.iloc[0]
        first_detection_mjd = first_detection["mjd"]

        features = []
        for band in self.bands:
            detections_band = detections[detections["fid"] == band]
            band_features = {
                "positive_fraction": np.nan,
                "n_forced_phot_band_before": np.nan,
                "dbrightness_first_det_band": np.nan,
                "dbrightness_forced_phot_band": np.nan,
                "last_brightness_before_band": np.nan,
                "max_brightness_before_band": np.nan,
                "median_brightness_before_band": np.nan,
                "n_forced_phot_band_after": np.nan,
                "max_brightness_after_band": np.nan,
                "median_brightness_after_band": np.nan,
            }
            if len(detections_band) >= self.detections_min_len:
                positive_fraction = np.mean(detections_band["brightness"].values > 0)
                band_features["positive_fraction"] = positive_fraction
                detections_band = detections_band.sort_values("mjd")
                first_brightness_in_band = detections_band.iloc[0]["brightness"]
            else:
                first_brightness_in_band = np.nan

            if not self.use_forced_photo or forced_photometry is None:
                features += self._dict_to_features(band_features, band)
                continue

            forced_photometry = forced_photometry[
                forced_photometry["unit"] == self.unit
            ]
            forced_photometry_band = forced_photometry[forced_photometry["fid"] == band]
            if len(forced_photometry_band) == 0:
                features += self._dict_to_features(band_features, band)
                continue

            forced_phot_band_before = forced_photometry_band[
                forced_photometry_band["mjd"] < first_detection_mjd
            ]
            forced_phot_band_before = forced_phot_band_before.sort_values("mjd")

            if len(forced_phot_band_before) > 0:
                band_features["n_forced_phot_band_before"] = len(
                    forced_phot_band_before
                )
                band_features["last_brightness_before_band"] = (
                    forced_phot_band_before.iloc[-1]["brightness"]
                )
                band_features["dbrightness_first_det_band"] = (
                    first_brightness_in_band
                    - band_features["last_brightness_before_band"]
                )
                band_features["dbrightness_forced_phot_band"] = (
                    first_brightness_in_band
                    - np.median(forced_phot_band_before["brightness"])
                )
                band_features["max_brightness_before_band"] = np.max(
                    forced_phot_band_before["brightness"]
                )
                band_features["median_brightness_before_band"] = np.median(
                    forced_phot_band_before["brightness"]
                )
            else:
                band_features["n_forced_phot_band_before"] = len(
                    forced_phot_band_before
                )
                band_features["last_brightness_before_band"] = np.nan
                band_features["dbrightness_first_det_band"] = np.nan
                band_features["dbrightness_forced_phot_band"] = np.nan
                band_features["max_brightness_before_band"] = np.nan
                band_features["median_brightness_before_band"] = np.nan

            forced_phot_band_after = forced_photometry_band[
                forced_photometry_band["mjd"] > first_detection_mjd
            ]

            band_features["n_forced_phot_band_after"] = len(forced_phot_band_after)
            if band_features["n_forced_phot_band_after"] == 0:
                features += self._dict_to_features(band_features, band)
                continue

            band_features["max_brightness_after_band"] = np.max(
                forced_phot_band_after["brightness"]
            )
            band_features["median_brightness_after_band"] = np.median(
                forced_phot_band_after["brightness"]
            )

            features += self._dict_to_features(band_features, band)

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

    def _dict_to_features(self, band_features: Dict[str, float], band: str) -> List:
        features = []
        for k, v in band_features.items():
            features.append((k, v, band))

        return features
