import numpy as np
import pandas as pd
from astropy import units as u
import extinction
from astropy.cosmology import WMAP5

import numba
from numba import jit

from scipy.optimize import minimize

import jax.numpy as jnp
from jax.nn import sigmoid as jax_sigmoid
from jax import grad
from jax import jit as jax_jit

from ..core.base import FeatureExtractor, AstroObject
from typing import List, Optional

import jax

jax.config.update("jax_enable_x64", True)


class SPMExtractor(FeatureExtractor):
    """Note: the firsts calls are really expensive because of jax compilations"""

    def __init__(
        self,
        bands: List[str],
        unit: str,
        redshift: Optional[str] = None,
        extinction_color_excess: Optional[str] = None,
        forced_phot_prelude: Optional[float] = None,
    ):

        self.version = "1.0.1"
        self.cws = {
            "u": 3671.0,
            "g": 4827.0,
            "r": 6223.0,
            "i": 7546.0,
            "z": 8691.0,
            "Y": 9712.0,
        }
        for band in bands:
            if band not in self.cws.keys():
                raise ValueError(f"{band} is not a valid band")

        self.bands = bands
        valid_units = ["diff_flux"]
        if unit not in valid_units:
            raise ValueError(f"{unit} is not a valid unit ({valid_units})")
        self.unit = unit
        self.redshift_name = redshift
        self.extinction_color_excess_name = extinction_color_excess
        self.forced_phot_prelude = forced_phot_prelude
        self.sn_model = SNModel(self.bands, debugging=False)

    def get_observations(self, astro_object: AstroObject) -> pd.DataFrame:
        observations = astro_object.detections.copy()
        mjd_first_detection = np.min(observations["mjd"])
        if astro_object.forced_photometry is not None:
            observations = pd.concat(
                [observations, astro_object.forced_photometry], axis=0
            )

        observations = observations[observations["unit"] == self.unit]
        observations = observations[observations["brightness"].notna()]
        if self.forced_phot_prelude is not None:
            before_prelude = mjd_first_detection - self.forced_phot_prelude
            observations = observations[observations["mjd"] > before_prelude]
        observations["mjd"] -= mjd_first_detection

        # Backwards compatibility: old SPM used mili Jansky, not uJy
        observations[["brightness", "e_brightness"]] *= 0.001
        return observations

    def _deattenuation_factor(self, band: str, mwebv: float):
        rv = 3.1
        av = rv * mwebv

        return 10 ** (
            extinction.odonnell94(np.array([self.cws[band]]), av, rv)[0] / 2.5
        )

    def _correct_lightcurve(
        self, times, fluxes, errors, bands, available_bands, mwebv, zhost
    ) -> None:

        # lightcurve corrections: dust, redshift
        if zhost is None or zhost < 0.003:
            zdeatt = 1.0
        else:
            zdeatt = 10.0 ** (
                -float((WMAP5.distmod(0.3) - WMAP5.distmod(zhost)) / u.mag) / 2.5
            )

        if mwebv is not None:
            for band in available_bands:
                deatt = self._deattenuation_factor(band, mwebv) * zdeatt
                band_mask = bands == band
                fluxes[band_mask] *= deatt
                errors[band_mask] *= deatt

        # times
        # times /= (1+zhost)

    def compute_features_single_object(self, astro_object: AstroObject):
        metadata = astro_object.metadata
        if self.redshift_name is None:
            redshift = None
        else:
            redshift = metadata[metadata["name"] == self.redshift_name]["value"].values[
                0
            ]
            redshift = float(redshift)

        if self.extinction_color_excess_name is None:
            extinction_color_excess = None
        else:
            extinction_color_excess = metadata[
                metadata["name"] == self.extinction_color_excess_name
            ]["value"].values[0]
            extinction_color_excess = float(extinction_color_excess)

        observations = self.get_observations(astro_object)

        times = observations["mjd"].values
        flux = observations["brightness"].values
        e_flux = observations["e_brightness"].values
        bands = observations["fid"].values
        available_bands = np.unique(bands)

        self._correct_lightcurve(
            times,
            flux,
            e_flux,
            bands,
            available_bands,
            extinction_color_excess,
            redshift,
        )

        self.sn_model.fit(times, flux, e_flux, bands)

        model_parameters = self.sn_model.get_model_parameters()
        chis = self.sn_model.get_chis()

        param_names = [
            "SPM_A",
            "SPM_t0",
            "SPM_gamma",
            "SPM_beta",
            "SPM_tau_rise",
            "SPM_tau_fall",
        ]
        features = []
        for band_index, band in enumerate(self.bands):
            for param_index, param_name in enumerate(param_names):
                model_param_index = band_index * len(param_names) + param_index
                features.append((param_name, model_parameters[model_param_index], band))

        for band, chi in zip(self.bands, chis):
            features.append(("SPM_chi", chi, band))

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


class SNModel:
    def __init__(self, bands: List[str], debugging: bool = False):
        self.parameters = None
        self.chis = None
        self.bands = bands
        self.debugging = debugging
        numba.set_num_threads(1)

    def fit(self, times, fluxpsf, obs_errors, bands):
        """Assumptions:
        min(times) == 0"""

        times = times.astype(np.float64)
        fluxpsf = fluxpsf.astype(np.float64)
        obs_errors = obs_errors.astype(np.float64)

        # Available bands
        available_bands = np.unique(bands)

        initial_guess = []
        bounds = []

        # Parameter guess
        tol = 1e-2

        prefered_order = list("irzYgu")
        for c in prefered_order:
            if c in available_bands:
                best_available_band = c
                break

        band_mask = bands == best_available_band
        t0_guess = times[band_mask][np.argmax(fluxpsf[band_mask])]
        t0_bounds = [-50.0, np.max(times)]
        t0_guess = np.clip(t0_guess - 10.0, t0_bounds[0] + tol, t0_bounds[1] - tol)
        gamma_guess = 14.0
        beta_guess = 0.5
        trise_guess = 7.0
        tfall_guess = 28.0

        for band in available_bands:
            band_mask = bands == band
            band_flux = fluxpsf[band_mask]

            # Parameter bounds
            max_band_flux = np.max(band_flux)

            A_bounds = [np.abs(max_band_flux) / 10.0, np.abs(max_band_flux) * 10.0]
            gamma_bounds = [1.0, 120.0]
            beta_bounds = [0.0, 1.0]
            trise_bounds = [1.0, 100.0]
            tfall_bounds = [1.0, 180.0]

            bounds += [
                A_bounds,
                t0_bounds,
                gamma_bounds,
                beta_bounds,
                trise_bounds,
                tfall_bounds,
            ]

            # Parameter guess
            A_guess = np.clip(
                1.2 * np.abs(max_band_flux), A_bounds[0] * 1.1, A_bounds[1] * 0.9
            )

            # reference guess
            p0 = [A_guess, t0_guess, gamma_guess, beta_guess, trise_guess, tfall_guess]

            initial_guess.append(p0)
        initial_guess = np.concatenate(initial_guess, axis=0)
        band_mapper = dict(zip("ugrizY", range(6)))

        # debugging
        # for g, b in zip(initial_guess, bounds):
        #     print(g, b)
        #     print(b[0] < g, g < b[1])

        negative_observations = (fluxpsf + obs_errors) < 0.0
        negative_observations = negative_observations.astype(np.float64)
        ignore_negative_fluxes = np.exp(
            -(
                ((fluxpsf + obs_errors) * negative_observations / (obs_errors + 1.0e-3))
                ** 2
            )
        )

        bands_num = np.vectorize(band_mapper.get)(bands)
        available_bands_num = np.vectorize(band_mapper.get)(available_bands)
        smooth_error = np.percentile(obs_errors, 10) * 0.5
        # smooth_error = smooth_error.astype(np.float32)
        # ignore_negative_fluxes = ignore_negative_fluxes.astype(np.float32)

        # padding
        pad_times = pad(times, np.min(times))
        pad_fluxpsf = pad(fluxpsf, 0.0)
        pad_obs_errors = pad(obs_errors, 1.0)
        pad_bands_num = pad(bands_num, -1.0)
        pad_ignore_negative_fluxes = pad(ignore_negative_fluxes, 0.0)

        res = minimize(
            objective_function_jax,
            initial_guess.astype(np.float64),
            jac=grad_objective_function_jax,
            args=(
                pad_times,
                pad_fluxpsf,
                pad_obs_errors,
                pad_bands_num,
                available_bands_num,
                smooth_error,
                pad_ignore_negative_fluxes,
            ),
            method="TNC",  # 'L-BFGS-B',
            bounds=bounds,
            options={"maxfun": 1000},  # {'iprint': -1, 'maxcor': 10, 'maxiter': 400}
        )

        # with open('objective_function_numba.txt', 'w') as f:
        #     objective_function.inspect_types(file=f)

        success = res.success
        best_params = res.x.reshape(-1, 6)

        parameters = []
        chis = []
        for band in self.bands:
            if band in available_bands:
                index = available_bands.tolist().index(band)
                parameters.append(best_params[index, :])

                band_mask = bands == band
                band_times = times[band_mask]
                band_flux = fluxpsf[band_mask]
                band_errors = obs_errors[band_mask]

                predictions = model_inference_stable(
                    band_times.astype(np.float64), *best_params[index, :]
                )

                if self.debugging:
                    import matplotlib.pyplot as plt

                    plt.scatter(band_times, band_flux)
                    order = np.argsort(band_times)
                    plt.plot(band_times[order], predictions[order])
                    plt.show()

                chi = np.sum(
                    (predictions - band_flux) ** 2 / (band_errors + 1.0e-3) ** 2
                )
                chi_den = len(predictions) - 6
                if chi_den >= 1:
                    chi_per_degree = chi / chi_den
                else:
                    chi_per_degree = np.NaN

                chis.append(chi_per_degree)
            else:
                parameters.append([np.nan] * 6)
                chis.append(np.nan)

        self.parameters = np.concatenate(parameters, axis=0)
        self.chis = np.array(chis)

    def get_model_parameters(self) -> List[float]:
        return self.parameters.tolist()

    def get_chis(self) -> List[float]:
        return self.chis.tolist()


def pad(x_array: np.ndarray, fill_value: float) -> np.ndarray:
    original_length = len(x_array)
    pad_length = 250 - (original_length % 250)
    pad_array = np.array([fill_value] * pad_length)
    return np.concatenate([x_array, pad_array])


@jax_jit
def objective_function_jax(
    params: np.ndarray,
    times: np.ndarray,
    fluxpsf: np.ndarray,
    obs_errors: np.ndarray,
    bands: np.ndarray,
    available_bands: np.ndarray,
    smooth_error: float,
    ignore_negative_fluxes,
) -> float:

    params = params.reshape(-1, 6)
    sum_sqerrs = 0.0

    for i, band in enumerate(available_bands):
        band_mask = bands == band
        band_times = times
        band_flux = fluxpsf
        band_errors = obs_errors
        band_ignore_negative_fluxes = ignore_negative_fluxes

        band_params = params[i]

        A = band_params[0]
        t0 = band_params[1]
        gamma = band_params[2]
        beta = band_params[3]
        t_rise = band_params[4]
        t_fall = band_params[5]

        sigmoid_factor = 1.0 / 2.0
        t1 = t0 + gamma

        sigmoid_exp_arg = sigmoid_factor * (band_times - t1)
        sig = jax_sigmoid(sigmoid_exp_arg)

        # push to 0 and 1
        sig *= sigmoid_exp_arg > -10.0
        sig = jnp.maximum(sig, (sigmoid_exp_arg >= 10.0))

        # if t_fall < t_rise, the output diverges for early times
        stable_case = t_fall < t_rise
        sig *= stable_case + (1 - stable_case) * (band_times > t1)

        den_exp_arg = (band_times - t0) / t_rise

        one_over_den = jax_sigmoid(den_exp_arg)

        # push to 0
        den_exp_arg_zero_mask = den_exp_arg > -20.0
        one_over_den *= den_exp_arg_zero_mask

        fall_exp_arg = -(band_times - t1) / t_fall
        fall_exp_arg = jnp.clip(fall_exp_arg, -20.0, 20.0)
        model_output = (
            (
                (1.0 - beta) * jnp.exp(fall_exp_arg) * sig
                + (1.0 - beta * (band_times - t0) / gamma) * (1.0 - sig)
            )
            * A
            * one_over_den
        )

        band_sqerr = ((model_output - band_flux) / (band_errors + smooth_error)) ** 2
        band_sqerr *= band_mask

        sum_sqerrs += jnp.dot(band_sqerr, band_ignore_negative_fluxes)
        # sum_sqerrs += np.sum(band_sqerr)

    params_var = jnp.array(
        [
            jnp.var(params[:, 0]) + 1,
            jnp.var(params[:, 1]) + 0.05,
            jnp.var(params[:, 2]) + 0.05,
            jnp.var(params[:, 3]) + 0.005,
            jnp.var(params[:, 4]) + 0.05,
            jnp.var(params[:, 5]) + 0.05,
        ]
    )

    lambdas = jnp.array(
        [
            0.0,  # A
            1.0,  # t0
            0.1,  # gamma
            20.0,  # beta
            0.7,  # t_rise
            0.01,  # t_fall
        ],
    )

    regularization = jnp.dot(lambdas, jnp.sqrt(params_var))
    # print(sum_sqerrs, regularization)
    # print(lambdas*params_var)
    loss = sum_sqerrs + regularization
    return loss


grad_objective_function_jax = jax_jit(grad(objective_function_jax))


@jit(nopython=True)
def model_inference_stable(times, A, t0, gamma, beta, t_rise, t_fall):
    sigmoid_factor = 1.0 / 2.0
    t1 = t0 + gamma

    sigmoid_exp_arg = -sigmoid_factor * (times - t1)
    sigmoid_exp_arg_big_mask = sigmoid_exp_arg >= 10.0
    sigmoid_exp_arg = np.clip(sigmoid_exp_arg, -10.0, 10.0)
    sigmoid = 1.0 / (1.0 + np.exp(sigmoid_exp_arg))
    sigmoid[sigmoid_exp_arg_big_mask] = 0.0

    # if t_fall < t_rise, the output diverges for early times
    if t_fall < t_rise:
        sigmoid_zero_mask = times < t1
        sigmoid[sigmoid_zero_mask] = 0.0

    den_exp_arg = -(times - t0) / t_rise
    den_exp_arg_big_mask = den_exp_arg >= 20.0

    den_exp_arg = np.clip(den_exp_arg, -20.0, 20.0)
    one_over_den = 1.0 / (1.0 + np.exp(den_exp_arg))
    one_over_den[den_exp_arg_big_mask] = 0.0

    fall_exp_arg = -(times - t1) / t_fall
    fall_exp_arg = np.clip(fall_exp_arg, -20.0, 20.0)
    model_output = (
        (
            (1.0 - beta) * np.exp(fall_exp_arg) * sigmoid
            + (1.0 - beta * (times - t0) / gamma) * (1.0 - sigmoid)
        )
        * A
        * one_over_den
    )

    return model_output
