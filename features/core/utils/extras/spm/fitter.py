import functools

import numpy as np
import pandas as pd
from scipy import optimize

from . import models, objective, guess
from .._utils import multiindex

_INDICES = (
    "SPM_A",
    "SPM_t0",
    "SPM_gamma",
    "SPM_beta",
    "SPM_tau_rise",
    "SPM_tau_fall",
    "SPM_chi",
)


@functools.lru_cache
def _indices_with_fid(fid: str) -> pd.MultiIndex:
    return multiindex(_INDICES, (fid,))


@functools.lru_cache
def _get_model(version: str):
    return getattr(models, version)


@functools.lru_cache
def _get_objective(version: str):
    return getattr(objective, version), getattr(objective, f"{version}_grad")


def single_band(
    version: str,
    time: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    alt: bool = False,
    bug: bool = False,
):
    time = time - np.min(time)

    initial, bounds = guess.single_band(time, flux, alt=alt, bug=bug)
    kwargs = dict(ftol=1e-8, sigma=error + 5) if alt else dict(ftol=initial[0] / 20)
    func = _get_model(version)

    try:
        params, *_ = optimize.curve_fit(
            func, time, flux, p0=initial, bounds=bounds, **kwargs
        )
    except (ValueError, RuntimeError, optimize.OptimizeWarning):
        try:
            kwargs.update(ftol=0.1 if alt else initial[0] / 3)
            params, *_ = optimize.curve_fit(
                func, time, flux, p0=initial, bounds=bounds, **kwargs
            )
        except (ValueError, RuntimeError, optimize.OptimizeWarning):
            params = np.full_like(initial, np.nan)

    prediction = func(time, *params)
    dof = prediction.size - params.size
    chi = (
        np.nan
        if dof < 1
        else np.sum((prediction - flux) ** 2 / (error + 0.01) ** 2) / dof
    )
    return pd.Series([*params, chi], index=_INDICES)


def multi_band(
    version: str,
    time: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    band: np.ndarray,
    preferred="irzYgu",
    mult=25,
):
    time = time - np.min(time)

    initial, bounds = guess.multi_band(time, flux, band, preferred=preferred)
    fids, band = np.unique(
        band, return_inverse=True
    )  # `band` now has the fids as index of fids
    smooth = np.percentile(error, 10) * 0.5

    weight = np.exp(-((flux + error) * ((flux + error) < 0) / (error + 1)) ** 2)

    # Padding is needed to minimize recompilations of jax jit functions
    time, flux, error, band, weight = _pad(time, flux, error, band, weight, mult)

    func = _get_model(version)
    obj, grad = _get_objective(version)

    args = (time, flux, error, band, np.arange(fids.size), smooth, weight)  # For objective function
    kwargs = dict(method="TNC", options={"maxfun": 1000})  # For minimizer
    result = optimize.minimize(
        obj, initial, jac=grad, args=args, bounds=bounds, **kwargs
    )

    n_params = 6
    params = result.x.reshape((-1, n_params))

    final = []
    for i, fid in enumerate(fids):
        mask = band == i

        prediction = func(time[mask], *params[i])

        dof = prediction.size - n_params
        chi = (
            np.nan
            if dof < 1
            else np.sum((prediction - flux[mask]) ** 2 / (error[mask] + 5) ** 2) / dof
        )
        final.append(pd.Series([*params[i], chi], index=_indices_with_fid(fid)))

    return pd.concat(final)


def _pad(time, flux, error, band, weight, mult=25):
    pad = (
        mult - time.size % mult
    )  # All padded arrays are assumed to have the same length
    time = np.pad(time, (0, pad), "constant", constant_values=(0, 0))
    flux = np.pad(flux, (0, pad), "constant", constant_values=(0, 0))
    band = np.pad(band, (0, pad), "constant", constant_values=(0, -1))
    error = np.pad(error, (0, pad), "constant", constant_values=(0, 1))
    weight = np.pad(weight, (0, pad), "constant", constant_values=(0, 0))
    return time, flux, error, band, weight
