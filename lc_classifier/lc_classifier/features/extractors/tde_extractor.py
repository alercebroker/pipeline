# Based on the work of Manuel Pavez

from ..core.base import FeatureExtractor, AstroObject
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import jax.numpy as jnp
from jax import jit as jax_jit

from typing import List

from ...utils import flux2mag, flux_err_2_mag_err

import jax

jax.config.update("jax_enable_x64", True)


class TDETailExtractor(FeatureExtractor):
    version = "1.0.1"
    unit = "diff_flux"

    def __init__(self, bands: List[str]):
        self.bands = bands

    def get_observations(self, astro_object: AstroObject) -> pd.DataFrame:
        observations = astro_object.detections
        if astro_object.forced_photometry is not None:
            observations = pd.concat(
                [observations, astro_object.forced_photometry], axis=0
            )
        observations = observations[observations["unit"] == self.unit]
        observations = observations[observations["brightness"].notna()]
        observations = observations[observations["e_brightness"] > 0.0]
        return observations

    def compute_features_single_object(self, astro_object: AstroObject):
        observations = self.get_observations(astro_object)

        diff_fluxes = observations[observations.unit == "diff_flux"].copy()
        diff_fluxes["brightness"] = np.abs(diff_fluxes["brightness"])
        diff_fluxes["e_brightness"] = flux_err_2_mag_err(
            diff_fluxes["e_brightness"], diff_fluxes["brightness"]
        )
        diff_fluxes["brightness"] = flux2mag(diff_fluxes["brightness"])
        diff_fluxes["unit"] = "diff_magnitude"
        observations = diff_fluxes
        observations = observations[observations["e_brightness"] < 1.0]
        observations = observations[observations["brightness"] < 30.0]

        features = []
        for band in self.bands:
            band_observations = observations[observations["fid"] == band]
            if len(band_observations) < 2:
                features.append(("TDE_decay", np.nan, band))
                features.append(("TDE_decay_chi", np.nan, band))
                features.append(("TDE_mag0", np.nan, band))
                continue

            brightest_obs = band_observations.sort_values("brightness").iloc[0]
            t_d = brightest_obs.mjd

            after_t_d = band_observations[
                (band_observations["mjd"] > t_d)
                & (band_observations["mjd"] < t_d + 200)
            ]

            x = 2.5 * np.log10(after_t_d.mjd.values - t_d + 40)
            y = after_t_d.brightness.values
            y_err = after_t_d.e_brightness.values + 1e-2

            omega = np.stack([np.ones(len(x)), x], axis=-1)
            inverr = 1.0 / y_err

            # weighted regularized linear regression
            w_a = inverr.reshape(-1, 1) * omega
            w_b = (y * inverr).reshape(-1, 1)
            coeffs = np.matmul(np.linalg.pinv(w_a), w_b).flatten()

            # Calculate reduced chi-squared statistic
            fitted_magnitude = coeffs[1] * x + coeffs[0]
            chi = np.sum((fitted_magnitude - y) ** 2 / y_err**2)
            chi_den = len(fitted_magnitude) - 2
            if chi_den >= 1:
                chi_per_degree = chi / chi_den
            else:
                chi_per_degree = np.nan

            features.append(("TDE_decay", coeffs[1], band))
            features.append(("TDE_decay_chi", chi_per_degree, band))
            features.append(("TDE_mag0", coeffs[0], band))

        features_df = pd.DataFrame(data=features, columns=["name", "value", "fid"])

        sids = astro_object.detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        features_df["sid"] = sid
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )


def pad(x_array: np.ndarray, fill_value: float) -> np.ndarray:
    original_length = len(x_array)
    pad_length = 25 - (original_length % 25)
    pad_array = np.array([fill_value] * pad_length)
    return np.concatenate([x_array, pad_array])


def fleet_model(t, a, w, m_0, t0):
    n_obs = len(t)
    t = pad(t, fill_value=0.0)
    func = fleet_model_jax(t, a, w, m_0, t0)
    func = np.array(func)[:n_obs]
    return func


@jax_jit
def fleet_model_jax(t, a, w, m_0, t0):
    t = t - t0
    func = jnp.exp(w * t) - a * w * t + m_0
    return func


class FleetExtractor(FeatureExtractor):
    version = "1.0.1"
    unit = "diff_flux"

    def __init__(self, bands: List[str]):
        self.bands = bands

    def get_observations(self, astro_object: AstroObject) -> pd.DataFrame:
        observations = astro_object.detections
        if astro_object.forced_photometry is not None:
            observations = pd.concat(
                [observations, astro_object.forced_photometry], axis=0
            )
        observations = observations[observations["unit"] == self.unit]
        observations = observations[observations["brightness"].notna()]
        observations = observations[
            observations["brightness"] > 1
        ]  # at least 1uJy positive detection
        observations = observations[observations["e_brightness"] > 0.0]
        return observations

    def compute_features_single_object(self, astro_object: AstroObject):
        observations = self.get_observations(astro_object)

        features = []
        for band in self.bands:
            band_observations = observations[observations["fid"] == band]
            if len(band_observations) < 4:
                features.append(("fleet_a", np.nan, band))
                features.append(("fleet_w", np.nan, band))
                features.append(("fleet_chi", np.nan, band))
                features.append(("fleet_m0", np.nan, band))
                features.append(("fleet_t0", np.nan, band))
                continue

            first_mjd = band_observations.sort_values("mjd").iloc[0]["mjd"]
            y = flux2mag(band_observations.brightness)
            y_err = (
                flux_err_2_mag_err(
                    band_observations.e_brightness, band_observations.brightness
                )
                + 1e-2
            )

            try:
                # noinspection PyTupleAssignmentBalance
                parameters, _ = curve_fit(
                    fleet_model,
                    band_observations["mjd"].values - first_mjd,
                    y,
                    sigma=y_err,
                    p0=[0.6, -0.05, np.mean(y), 0],
                    bounds=([0.0, -100.0, 0, -50], [10, 0, 30, 10000]),
                )

                model_prediction = fleet_model(
                    band_observations["mjd"].values - first_mjd, *parameters
                )

                chi = np.sum((model_prediction - y) ** 2 / y_err**2)
                chi_den = len(model_prediction) - 4
                if chi_den >= 1:
                    chi_per_degree = chi / chi_den
                else:
                    chi_per_degree = np.nan

                features.append(("fleet_a", parameters[0], band))
                features.append(("fleet_w", parameters[1], band))
                features.append(("fleet_chi", chi_per_degree, band))
                features.append(("fleet_m0", parameters[2], band))
                features.append(("fleet_t0", parameters[3], band))
            except RuntimeError:
                features.append(("fleet_a", np.nan, band))
                features.append(("fleet_w", np.nan, band))
                features.append(("fleet_chi", np.nan, band))
                features.append(("fleet_m0", np.nan, band))
                features.append(("fleet_t0", np.nan, band))

        features_df = pd.DataFrame(data=features, columns=["name", "value", "fid"])

        sids = astro_object.detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        features_df["sid"] = sid
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )


class ColorVariationExtractor(FeatureExtractor):
    version = "1.0.1"
    unit = "diff_flux"

    def __init__(self, window_len: float, band_1: str, band_2: str):
        self.window_len = window_len
        self.band_1 = band_1
        self.band_2 = band_2

    def get_observations(self, astro_object: AstroObject) -> pd.DataFrame:
        observations = astro_object.detections
        if astro_object.forced_photometry is not None:
            observations = pd.concat(
                [observations, astro_object.forced_photometry], axis=0
            )
        observations = observations[observations["unit"] == self.unit]
        observations = observations[observations["brightness"].notna()]
        observations = observations[observations["e_brightness"] > 0.0]
        observations = observations[
            observations["fid"].isin([self.band_1, self.band_2])
        ]

        return observations

    def compute_features_single_object(self, astro_object: AstroObject):
        observations = self.get_observations(astro_object).copy()

        diff_fluxes = observations[observations.unit == "diff_flux"].copy()
        diff_fluxes["brightness"] = np.abs(diff_fluxes["brightness"])
        diff_fluxes["e_brightness"] = flux_err_2_mag_err(
            diff_fluxes["e_brightness"], diff_fluxes["brightness"]
        )
        diff_fluxes["brightness"] = flux2mag(diff_fluxes["brightness"])
        diff_fluxes["unit"] = "diff_magnitude"
        observations = diff_fluxes
        observations = observations[observations["e_brightness"] < 1.0]
        observations = observations[observations["brightness"] < 30.0]

        observations["mjd"] -= observations["mjd"].min()
        observations["window"] = (observations["mjd"] // self.window_len).astype(int)

        def compute_color(df):
            fid_count = df[["fid", "brightness"]].groupby("fid").count()

            if len(fid_count) < 2:
                return

            if not np.all(fid_count.values >= 3):
                return

            fid_means = df[["fid", "brightness"]].groupby("fid").mean()

            color = fid_means.loc[self.band_1] - fid_means.loc[self.band_2]
            return color

        # pandas 2.2 and higher should use the include_groups=False arg
        window_colors = observations.groupby("window").apply(
            compute_color  # , include_groups=False
        )
        window_colors.dropna(inplace=True)

        if len(window_colors) > 1:
            color_std = np.std(window_colors.values, ddof=1)
        else:
            color_std = np.nan

        features_df = pd.DataFrame(
            data=[["color_variation", color_std, f"{self.band_1},{self.band_2}"]],
            columns=["name", "value", "fid"],
        )

        sids = astro_object.detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        features_df["sid"] = sid
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )
