# Based on the work of Vicente Pedreros

from ..core.base import FeatureExtractor, AstroObject
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import jax.numpy as jnp
from jax import jit as jax_jit

from typing import List

import jax

jax.config.update("jax_enable_x64", True)


def pad(x_array: np.ndarray, fill_value: float) -> np.ndarray:
    original_length = len(x_array)
    pad_length = 25 - (original_length % 25)
    pad_array = np.array([fill_value] * pad_length)
    return np.concatenate([x_array, pad_array])


def ulens_model(t, u0, tE, fs, t0, mag_0):
    n_obs = len(t)
    t = pad(t, fill_value=0.0)
    func = ulens_model_jax(t, u0, tE, fs, t0, mag_0)
    func = np.array(func)[:n_obs]
    return func


@jax_jit
def ulens_model_jax(t, u0, tE, fs, t0, mag_0):
    t = t - t0
    u = jnp.sqrt(u0**2 + (t / tE) ** 2)
    A = (u**2 + 2.0) / (u * jnp.sqrt(u**2 + 4))
    func = -2.5 * jnp.log10(fs * (A - 1) + 1) + mag_0
    return func


class MicroLensExtractor(FeatureExtractor):
    version = "1.0.1"
    unit = "magnitude"

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
        observations = observations[observations["e_brightness"] < 1.0]
        observations["mjd"] -= np.min(observations["mjd"])
        return observations

    def compute_features_single_object(self, astro_object: AstroObject):
        observations = self.get_observations(astro_object)

        features = []
        for band in self.bands:
            band_observations = observations[observations["fid"] == band]
            if len(band_observations) < 4:
                features.append(("ulens_u0", np.nan, band))
                features.append(("ulens_tE", np.nan, band))
                features.append(("ulens_fs", np.nan, band))
                features.append(("ulens_chi", np.nan, band))
                features.append(("ulens_t0", np.nan, band))
                features.append(("ulens_mag0", np.nan, band))
                continue

            mjd_max_flux = band_observations.sort_values("brightness").iloc[0]["mjd"]
            y = band_observations.brightness
            y_err = band_observations.e_brightness + 1e-2

            try:
                # noinspection PyTupleAssignmentBalance
                parameters, _ = curve_fit(
                    ulens_model,
                    band_observations["mjd"],
                    y,
                    sigma=y_err,
                    p0=[0.6, 20.0, 0.5, mjd_max_flux, np.median(y)],
                    bounds=(
                        [0, 0, 0, -np.inf, -np.inf],
                        [np.inf, np.inf, 1, np.inf, np.inf],
                    ),
                )

                model_prediction = ulens_model(band_observations["mjd"], *parameters)

                chi = np.sum((model_prediction - y) ** 2 / y_err**2)
                chi_den = len(model_prediction) - 4
                if chi_den >= 1:
                    chi_per_degree = chi / chi_den
                else:
                    chi_per_degree = np.nan

                features.append(("ulens_u0", parameters[0], band))
                features.append(("ulens_tE", parameters[1], band))
                features.append(("ulens_fs", parameters[2], band))
                features.append(("ulens_chi", chi_per_degree, band))
                features.append(("ulens_t0", parameters[3], band))
                features.append(("ulens_mag0", parameters[4], band))
            except RuntimeError:
                features.append(("ulens_u0", np.nan, band))
                features.append(("ulens_tE", np.nan, band))
                features.append(("ulens_fs", np.nan, band))
                features.append(("ulens_chi", np.nan, band))
                features.append(("ulens_t0", np.nan, band))
                features.append(("ulens_mag0", np.nan, band))

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
