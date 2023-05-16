import functools

import numpy as np
import pandas as pd
from jax import grad as jgrad
from jax import jit as jjit
from jax import numpy as jnp
from jax.nn import sigmoid as jsigmoid
from numba import njit
from scipy import optimize

from ._utils import multiindex

_INDICES = ("SPM_A", "SPM_t0", "SPM_gamma", "SPM_beta", "SPM_tau_raise", "SPM_tau_fall", "SPM_chi")


@functools.lru_cache
def _indices_with_fid(fid: str) -> pd.MultiIndex:
    return multiindex(_INDICES, (fid,))


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

    sigmoid_arg = sigmoid_factor * (times - t1)
    sigmoid = jsigmoid(jnp.clip(sigmoid_arg, -10, 10))
    sigmoid *= sigmoid_arg > -10  # set to zero  below -10
    sigmoid = jnp.where(sigmoid_arg < 10, sigmoid, 1)  # set to one above 10

    stable = t_fall >= t_rise  # Early times diverge in unstable case
    sigmoid *= stable + (1 - stable) * (times > t1)

    raise_arg = -(times - t0) / t_rise
    den = 1 + jnp.exp(jnp.clip(raise_arg, -20, 20))

    fall_arg = jnp.clip(-(times - t1) / t_fall, -20, 20)
    temp = (1 - beta) * jnp.exp(fall_arg) * sigmoid + (1 - beta * (times - t0) / gamma) * (1 - sigmoid)
    return jnp.where(raise_arg > 20, temp * ampl / den, 0)


@jjit
def _spm_v2_loss(params, time, flux, error, band, fids, smooth):
    negative = (flux + error) < 0
    # Give lower weights in square error to negative detections, based on how negative it is
    weight = jnp.exp(-((flux + error) * negative / (error + 1)) ** 2)

    params = params.reshape((-1, 6))
    sq_err = 0.0

    for i in fids:
        ampl, t0, gamma, beta, t_rise, t_fall = params[i]

        sim = _spm_v2(time, ampl, t0, gamma, beta, t_rise, t_fall)
        sq_err_i = ((sim - flux) / (error + smooth)) ** 2

        sq_err += jnp.dot(sq_err_i * (band == i), weight)

    var = jnp.var(params, axis=0) + jnp.array([1, 5e-2, 5e-2, 5e-3, 5e-2, 5e-2])
    lambdas = jnp.array([0, 1, 0.1, 20, 0.7, 0.01])

    regularization = jnp.dot(lambdas, jnp.sqrt(var))
    return regularization + sq_err


_grad_spm_v2_loss = jjit(jgrad(_spm_v2_loss))


def _guess_and_bounds_v2(time, flux, band, preferred):
    for fid in preferred:
        if fid in band:
            ref = fid
            break
    else:
        raise ValueError(f"None of bands {preferred} found in the provided bands")

    tol = 1e-2
    mask = band == ref
    t0_bounds = [-50, np.max(time)]
    t0_guess = time[mask][np.argmax(flux[mask])] - 10
    # Order: t0, gamma, beta, t_rise, t_fall
    static_bounds = [t0_bounds, [1, 120], [0, 1], [1, 100], [1, 180]]
    static_guess = [np.clip(t0_guess, t0_bounds[0] + tol, t0_bounds[1] - tol), 14, 0.5, 7, 28]

    guess, bounds = [], []
    for fid in np.unique(band):
        mask = band == fid

        fmax = np.max(flux[mask])
        ampl_bounds = [np.abs(fmax) / 10., np.abs(fmax) * 10.]
        ampl_guess = np.clip(1.2 * fmax, ampl_bounds[0] * 1.1, ampl_bounds[1] * 0.9)

        bounds += [ampl_bounds] + static_bounds
        guess += [ampl_guess] + static_guess

    return np.array(guess, dtype=np.float32), np.array(bounds, dtype=np.float32)


def _pad(time, flux, error, band=None, multiple=25):
    pad = multiple - time.size % multiple  # All padded arrays are assumed to have the same length
    time = np.pad(time, (0, pad), "constant", constant_values=(0, 0))
    flux = np.pad(flux, (0, pad), "constant", constant_values=(0, 0))
    band = np.pad(band, (0, pad), "constant", constant_values=(0, -1)) if band is not None else band
    error = np.pad(error, (0, pad), "constant", constant_values=(0, 1))
    return time, flux, error, band


def fit_spm_v2(time, flux, error, band, preferred="irzYgu", multiple=25):
    time, flux, error = time.astype(np.float32), flux.astype(np.float32), error.astype(np.float32)

    time = time - np.min(time)
    guess, bounds = _guess_and_bounds_v2(time, flux, band, preferred)
    smooth = np.percentile(error, 10) * 0.5

    fids, band = np.unique(band, return_inverse=True)
    ifids = np.arange(fids.size)

    # Padding is needed to minimize recompilations of jax jit functions
    time, flux, error, band = _pad(time, flux, error, band, multiple)

    args = (time, flux, error, band, ifids, smooth)
    kwargs = dict(method="TNC", options={"maxfun": 1000})
    result = optimize.minimize(_spm_v2_loss, guess, jac=_grad_spm_v2_loss, args=args, bounds=bounds, **kwargs)
    params = result.x.reshape((-1, 6))

    final = []
    for i, fid in enumerate(fids):
        mask = band == i

        prediction = _spm_v2(time[mask], *params[i])
        dof = prediction.size - params[i].size
        chi = np.nan if dof < 1 else np.sum((prediction - flux[mask]) ** 2 / (error[mask] + 5) ** 2) / dof
        final.append(pd.Series([*params[i], chi], index=_indices_with_fid(fid)))

    return pd.concat(final)


def fit_spm_v1(time: np.ndarray, flux: np.ndarray, error: np.ndarray, alt: bool = False) -> pd.Series:
    time, flux, error = time.astype(np.float32), flux.astype(np.float32), error.astype(np.float32)
    time = time - np.min(time)
    imax = np.argmax(flux)

    fmax = flux[imax]
    # order for bounds/guess: amplitude, t0, gamma, beta, t_rise, t_fall (lower first, upper second)
    if alt:
        bounds = [[fmax / 3, -50, 1, 0, 1, 1], [fmax * 3, 70, 100, 1, 100, 100]]
        guess = np.clip([1.2 * fmax, time[imax] * 2 / 3, time[imax], 0.5, time[imax] / 2, 50], *bounds)
        kwargs = dict(ftol=1e-8, sigma=error + 5)
    else:
        bounds = [[fmax / 3, -50, 1, 0, 1, 1], [fmax * 3, 50, 100, 1, 100, 100]]
        # TODO: 3 * fmax should be 1.2 * fmax, but model is trained with the bug. DO NOT FIX UNTIL RETRAINED!!
        guess = np.clip([3 * fmax, -5, np.max(time), 0.5, time[imax] / 2, 40], *bounds)
        kwargs = dict(ftol=guess[0] / 20)

    try:
        params, *_ = optimize.curve_fit(_spm_v1, time, flux, p0=guess, bounds=bounds, **kwargs)
    except (ValueError, RuntimeError, optimize.OptimizeWarning):
        try:
            kwargs.update(ftol=0.1 if alt else guess[0] / 3)
            params, *_ = optimize.curve_fit(_spm_v1, time, flux, p0=guess, bounds=bounds, **kwargs)
        except (ValueError, RuntimeError, optimize.OptimizeWarning):
            params = np.full_like(guess, np.nan)

    prediction = _spm_v1(time, *params)
    dof = prediction.size - params.size
    chi = np.nan if dof < 1 else np.sum((prediction - flux) ** 2 / (error + 0.01) ** 2) / dof
    return pd.Series([*params, chi], index=_INDICES)
