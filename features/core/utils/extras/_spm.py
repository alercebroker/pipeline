import functools

import pandas as pd

from . import mag2flux, deattenuate
from .spm import fitter


@functools.lru_cache
def _get_fitter(multiband: bool):
    return getattr(fitter, "multi_band" if multiband else "single_band")


@functools.lru_cache
def _get_conversion(sid: str):
    return getattr(mag2flux, sid.lower())


@functools.lru_cache
def _get_correction(sid: str):
    return getattr(deattenuate, sid.lower())


def fit_spm(
    df: pd.DataFrame,
    version: str,
    ml: bool = True,
    flux: bool = False,
    multiband: bool = False,
    correct: bool = False,
    **kwargs,
) -> pd.Series:
    """Fits supernova parametric model and returns the parameters. Only works for a single object at a time.

    Args:
        df: Frame with magnitudes, errors and times. Expects a single object and band at a time
        version: Model version to use (at the moment, either `v1` or `v2`)
        ml: Whether to use corrected magnitudes instead of uncorrected
        flux: Whether the "magnitudes" are actually in flux units
        multiband: Fit all bands simultaneously. Do not use with `apply` if grouping includes `fid`
        correct: Applies dust and redshift correct (requires fields `mwebv` and `z_final` in `df`)
        kwargs: Passed to the fitter function (available parameters depend on whether `multiband` is used)

    Returns:
        pd.Series: Fitted parameters
    """
    suffix = "_ml" if ml else ""
    mag, e_mag, mjd = df[[f"mag{suffix}", f"e_mag{suffix}", "mjd"]].T.values
    band = df["fid"].values

    target = mag  # Assumes fluxes or will be fixed below
    error = e_mag
    if not flux:
        for sid in df["sid"].unique():
            mask = df["sid"] == sid
            func = _get_conversion(sid)

            target[mask] = func(mag[mask])
            error[mask] = func(mag[mask] - e_mag[mask]) - target[mask]

    if correct:
        for sid in df["sid"].unique():
            mask = df["sid"] == sid
            func = _get_correction(sid)

            mwebv = df["mwebv"][mask].median()
            zhost = df["z_final"][mask].median()

            func(target[mask], error[mask], band[mask], mwebv, zhost)

    func = _get_fitter(multiband)
    commons = (version, mjd, target, error)

    return func(*commons, band, **kwargs) if multiband else func(*commons, **kwargs)
