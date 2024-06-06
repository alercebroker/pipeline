import pandas as pd

from ..core.base import FeatureExtractor, AstroObject
from ...utils import flux2mag, flux_err_2_mag_err
import numpy as np
from typing import List, Tuple


class ColorFeatureExtractor(FeatureExtractor):
    def __init__(self, bands: List[str], just_flux: bool):
        self.version = "1.0.0"
        self.bands = bands
        self.just_flux = just_flux

    def preprocess_detections_just_magnitude(
        self, detections: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        detections = detections[detections["brightness"].notna()]
        corrected_mags = detections[detections.unit == "magnitude"]
        corrected_mags = corrected_mags[corrected_mags["e_brightness"] < 1.0]

        diff_fluxes = detections[detections.unit == "diff_flux"].copy()
        diff_fluxes["brightness"] = np.abs(diff_fluxes["brightness"])
        diff_fluxes["e_brightness"] = flux_err_2_mag_err(
            diff_fluxes["e_brightness"], diff_fluxes["brightness"]
        )
        diff_fluxes["brightness"] = flux2mag(diff_fluxes["brightness"])
        diff_fluxes["unit"] = "diff_magnitude"
        diff_magnitudes = diff_fluxes

        return corrected_mags, diff_magnitudes

    def preprocess_detections_just_flux(self, detections: pd.DataFrame) -> pd.DataFrame:
        detections = detections[detections["brightness"].notna()]
        detections = detections[detections["unit"] == "diff_flux"]
        return detections

    def compute_features_single_object(self, astro_object: AstroObject):
        detections = astro_object.detections

        fid = ",".join(self.bands)

        sids = detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        if self.just_flux:
            detections = self.preprocess_detections_just_flux(detections)
            features = self._diff_flux_colors(detections)
        else:
            corrected_mags, diff_magnitudes = self.preprocess_detections_just_magnitude(
                detections
            )
            corr_colors = self._magnitude_colors(corrected_mags, feature_suffix="_corr")
            colors = self._magnitude_colors(diff_magnitudes)
            features = colors + corr_colors

        features_df = pd.DataFrame(data=features, columns=["name", "value"])

        features_df["fid"] = fid
        features_df["sid"] = sid
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )

    def _diff_flux_colors(self, detections: pd.DataFrame) -> List[Tuple[str, float]]:
        # compute mean and max of each band
        band_p90_list = []
        for band in self.bands:
            band_detections = detections[detections["fid"] == band]
            if len(band_detections) == 0:
                band_p90_list.append(np.nan)
            else:
                band_flux_abs = np.abs(band_detections["brightness"])
                band_p90 = np.percentile(band_flux_abs, 90)
                band_p90_list.append(band_p90)

        features = []
        for i in range(len(self.bands) - 1):
            feature_name = f"{self.bands[i]}-{self.bands[i+1]}"
            feature_value = band_p90_list[i] / (band_p90_list[i + 1] + 1)
            features.append((feature_name, feature_value))

        return features

    def _magnitude_colors(
        self, detections: pd.DataFrame, feature_suffix: str = ""
    ) -> List[Tuple[str, float]]:
        # compute mean and max of each band
        band_means = []
        band_maxima = []
        for band in self.bands:
            band_detections = detections[detections["fid"] == band]
            if len(band_detections) == 0:
                band_means.append(np.nan)
                band_maxima.append(np.nan)
            else:
                band_mean = np.mean(band_detections["brightness"])
                band_means.append(band_mean)

                # max brightness, min magnitude
                band_max = np.min(band_detections["brightness"])
                band_maxima.append(band_max)

        features = []
        for i in range(len(self.bands) - 1):
            feature_name = f"{self.bands[i]}-{self.bands[i+1]}_mean" + feature_suffix
            feature_value = band_means[i] - band_means[i + 1]
            features.append([feature_name, feature_value])

            feature_name = f"{self.bands[i]}-{self.bands[i+1]}_max" + feature_suffix
            feature_value = band_maxima[i] - band_maxima[i + 1]
            features.append((feature_name, feature_value))

        return features
