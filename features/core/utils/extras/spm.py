import numpy as np
import pandas as pd
from jax import jit as jjit
from jax import numpy as jnp
from numba import njit
from scipy import optimize


@njit
def _spm_v1(times, ampl, t0, gamma, beta, t_rise, t_fall):
    """Direct usage of the model"""
    sigmoid_factor = 1 / 3
    t1 = t0 + gamma

    sigmoid = 1 / (1 + np.exp(-sigmoid_factor * (times - t1)))
    den = 1 + np.exp(-(times - t0) / t_rise)
    temp = (1 - beta) * np.exp(-(times - t1) / t_fall) * sigmoid + (1. - beta * (times - t0) / gamma) * (1 - sigmoid)
    return temp * ampl / den


@jjit
def _spm_v2(times, ampl, t0, gamma, beta, t_rise, t_fall):
    """Uses constrains to provide higher stability with respect to v1"""
    sigmoid_factor = 1 / 2
    t1 = t0 + gamma

    sigmoid_arg = -sigmoid_factor * (times - t1)
    sigmoid = 1 / (1 + jnp.exp(jnp.clip(sigmoid_arg, -10, 10)))
    sigmoid *= sigmoid_arg < 10  # Apply mask

    stable = t_fall >= t_rise  # Early times diverge in unstable case
    sigmoid *= stable + (1 - stable) * (times > t1)

    raise_arg = -(times - t0) / t_rise
    den = 1 / (1 + jnp.exp(jnp.clip(raise_arg, -20, 20)))

    fall_arg = jnp.clip(-(times - t1) / t_fall, -20, 20)
    temp = (1 - beta) * jnp.exp(fall_arg) * sigmoid + (1 - beta * (times - t0) / gamma) * (1 - sigmoid)
    return jnp.where(raise_arg < 20, temp * ampl / den, 0)


def fit_spm_v1(time: np.ndarray, flux: np.ndarray, error: np.ndarray) -> pd.Series:
    time = time - np.min(time)
    imax = np.argmax(flux)

    fmax = flux[imax]
    # order: amplitude, t0, gamma, beta, t_rise, t_fall (lower first, upper second)
    bounds = [[fmax / 3, -50, 1, 0, 1, 1], [fmax * 3, 50, 100, 1, 100, 100]]
    # TODO: 3 * fmax should be 1.2 * fmax, but model is trained with the bug. DO NOT FIX UNTIL RETRAINED!!
    guess = np.clip([3 * fmax, -5, np.max(time), 0.5, time[imax] / 2, 40], *bounds)

    try:
        params, *_ = optimize.curve_fit(_spm_v1, time, flux, p0=guess, bounds=bounds, ftol=guess[0] / 20)
    except (ValueError, RuntimeError, optimize.OptimizeWarning):
        try:
            params, *_ = optimize.curve_fit(_spm_v1, time, flux, p0=guess, bounds=bounds, ftol=guess[0] / 3)
        except (ValueError, RuntimeError, optimize.OptimizeWarning):
            params = np.full_like(guess, np.nan)

    prediction = _spm_v1(time, *params)
    n_dof = prediction.size - params.size
    chi_dof = np.nan if n_dof < 1 else np.sum((prediction - flux) ** 2 / (error + 0.01) ** 2) / n_dof
    return pd.Series([*params, chi_dof], index=["A", "t0", "gamma", "beta", "tau_raise", "tau_fall", "chi"])


def fit_spm_v1_alt(time: np.ndarray, flux: np.ndarray, error: np.ndarray) -> pd.Series:
    """Same as `fit_spm_v1`, but with different bounds, initial guess and tolerance"""
    time = time - np.min(time)
    imax = np.argmax(flux)

    fmax = flux[imax]
    # order: amplitude, t0, gamma, beta, t_rise, t_fall (lower first, upper second)
    bounds = [[fmax / 3, -50, 1, 0, 1, 1], [fmax * 3, 70, 100, 1, 100, 100]]
    guess = np.clip([1.2 * fmax, time[imax] * 2 / 3, time[imax], 0.5, time[imax] / 2, 50], *bounds)

    try:
        params, *_ = optimize.curve_fit(_spm_v1, time, flux, p0=guess, bounds=bounds, sigma=5 + error, ftol=1e-8)
    except (ValueError, RuntimeError, optimize.OptimizeWarning):
        try:
            params, *_ = optimize.curve_fit(_spm_v1, time, flux, p0=guess, bounds=bounds, sigma=5 + error, ftol=0.1)
        except (ValueError, RuntimeError, optimize.OptimizeWarning):
            params = np.full_like(guess, np.nan)

    prediction = _spm_v1(time, *params)
    n_dof = prediction.size - params.size
    chi_dof = np.nan if n_dof < 1 else np.sum((prediction - flux) ** 2 / (error + 0.01) ** 2) / n_dof
    return pd.Series([*params, chi_dof], index=["A", "t0", "gamma", "beta", "tau_raise", "tau_fall", "chi"])
