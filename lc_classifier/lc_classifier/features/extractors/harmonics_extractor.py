from ..core.base import FeatureExtractor, AstroObject
import numpy as np
import pandas as pd
from typing import List


class HarmonicsExtractor(FeatureExtractor):
    def __init__(
        self, bands: List[str], unit: str, use_forced_photo: bool, n_harmonics: int = 7
    ):

        self.version = "1.0.0"
        self.bands = bands
        valid_units = ["magnitude", "diff_flux"]
        if unit not in valid_units:
            raise ValueError(f"{unit} is not a valid unit ({valid_units})")
        self.unit = unit
        self.use_forced_photo = use_forced_photo
        self.n_harmonics = n_harmonics
        self.degrees_of_freedom = self.n_harmonics * 2 + 1
        if unit == "magnitude":
            self.error_tol = 1e-2
        elif unit == "diff_flux":
            self.error_tol = 1e-3

    def get_observations(self, astro_object: AstroObject) -> pd.DataFrame:
        observations = astro_object.detections
        if self.use_forced_photo:
            if astro_object.forced_photometry is not None:
                observations = pd.concat(
                    [observations, astro_object.forced_photometry], axis=0
                )
        observations = observations[observations["unit"] == self.unit]
        observations = observations[observations["brightness"].notna()]
        observations = observations[observations["e_brightness"] > 0.0]
        return observations

    def compute_features_single_object(self, astro_object: AstroObject):
        period_feature_name = "Multiband_period"
        if period_feature_name not in astro_object.features["name"].values:
            raise Exception("Harmonics extractor was not provided with period data")
        period = astro_object.features[
            astro_object.features["name"] == period_feature_name
        ]
        period = period["value"][0]

        observations = self.get_observations(astro_object)

        for band in self.bands:
            band_observations = observations[observations["fid"] == band]
            if np.isnan(period) or len(band_observations) < self.degrees_of_freedom:
                self._add_harmonic_info_to_astroobject(
                    astro_object,
                    [np.nan] * self.n_harmonics,
                    [np.nan] * (self.n_harmonics - 1),
                    np.nan,
                    np.nan,
                    band,
                )

                continue

            time = band_observations["mjd"].values
            brightness = band_observations["brightness"].values
            error = band_observations["e_brightness"].values + 10**-2

            best_freq = 1 / period

            omega = [np.array([[1.0] * len(time)])]
            timefreq = (
                2.0 * np.pi * best_freq * np.arange(1, self.n_harmonics + 1)
            ).reshape(1, -1).T * time
            omega.append(np.cos(timefreq))
            omega.append(np.sin(timefreq))

            # Omega.shape == (lc_length, 1+2*self.n_harmonics)
            omega = np.concatenate(omega, axis=0).T

            inverr = 1.0 / error

            # weighted regularized linear regression
            w_a = inverr.reshape(-1, 1) * omega
            w_b = (brightness * inverr).reshape(-1, 1)
            coeffs = np.matmul(np.linalg.pinv(w_a), w_b).flatten()
            fitted_magnitude = np.dot(omega, coeffs)
            coef_cos = coeffs[1 : self.n_harmonics + 1]
            coef_sin = coeffs[self.n_harmonics + 1 :]
            coef_mag = np.sqrt(coef_cos**2 + coef_sin**2)
            coef_phi = np.arctan2(coef_sin, coef_cos)

            # Relative phase
            coef_phi = coef_phi - coef_phi[0] * np.arange(1, self.n_harmonics + 1)
            coef_phi = coef_phi[1:] % (2 * np.pi)

            mse = np.mean((fitted_magnitude - brightness) ** 2)

            # Calculate reduced chi-squared statistic
            chi = np.sum(
                (fitted_magnitude - brightness) ** 2 / (error + self.error_tol) ** 2
            )
            chi_den = len(fitted_magnitude) - (1 + 2 * self.n_harmonics)
            if chi_den >= 1:
                chi_per_degree = chi / chi_den
            else:
                chi_per_degree = np.nan

            self._add_harmonic_info_to_astroobject(
                astro_object, coef_mag, coef_phi, mse, chi_per_degree, band
            )

    def _add_harmonic_info_to_astroobject(
        self,
        astro_object: AstroObject,
        harmonics_mag_list: List[float],
        harmonics_phase_list: List[float],
        harmonics_mse: float,
        harmonics_chi: float,
        band: str,
    ):

        features = [("Harmonics_mse", harmonics_mse), ("Harmonics_chi", harmonics_chi)]

        for i in range(self.n_harmonics):
            harmonic_index = i + 1
            features.append((f"Harmonics_mag_{harmonic_index}", harmonics_mag_list[i]))
            if harmonic_index == 1:
                continue
            features.append(
                (f"Harmonics_phase_{harmonic_index}", harmonics_phase_list[i - 1])
            )

        features_df = pd.DataFrame(data=features, columns=["name", "value"])

        sids = astro_object.detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        features_df["fid"] = band
        features_df["sid"] = sid
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )
