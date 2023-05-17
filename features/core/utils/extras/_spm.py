import functools
from typing import Any, Callable

import extinction
import numpy as np
import pandas as pd
from astropy.cosmology import WMAP5

from .spm import fitter

_CALLABLE_WITH_BANDS = Callable[[np.ndarray, np.ndarray, np.ndarray, Any], pd.Series]
_CALLABLE_WITHOUT_BANDS = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any], pd.Series]


@functools.lru_cache
def _get_fitter(multiband: bool):
    return getattr(fitter, "multi_band" if multiband else "single_band")


def _mag2flux_ztf(mag: np.ndarray) -> np.ndarray:
    """Magnitude to flux conversion for ZTF.

    Args:
        mag: Magnitudes

    Returns:
        np.ndarray: Fluxes
    """
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)


def _deattenuate_lsst(flux: np.ndarray, error: np.ndarray, band: np.ndarray, mwebv: float, zhost: float):
    """Modifies flux and error inplace"""
    zhost = zhost if zhost >= 0.003 else 0
    z_deatt = 10 ** (-(WMAP5.distmod(0.3).to("mag").value - WMAP5.distmod(zhost).to("mag")) / 2.5) if zhost else 1

    rv = 3.1
    av = rv * mwebv

    wavelengths = {  # Wavelength (in Angstroms) for each band
        "u": 3671.,
        "g": 4827.,
        "r": 6223.,
        "i": 7546.,
        "z": 8691.,
        "Y": 9712.,
    }

    for fid in np.unique(band):
        mask = band == fid
        dust_deatt, = 10 ** (extinction.odonnell94(np.full(1, wavelengths[band]), av, rv) / 2.5)

        flux[mask] *= z_deatt * dust_deatt
        error[mask] *= z_deatt * dust_deatt


def fit_spm(
    df: pd.DataFrame,
    version: str,
    ml: bool = True,
    flux: bool = False,
    multiband: bool = False,
    deattenuate: bool = False,
    **kwargs
) -> pd.Series:
    """Fits supernova parametric model and returns the parameters. Only works for a single object at a time.

    Args:
        df: Frame with magnitudes, errors and times. Expects a single object and band at a time
        version: Model version to use (at the moment, either `v1` or `v2`)
        ml: Whether to use corrected magnitudes instead of uncorrected
        flux: Whether the "magnitudes" are actually in flux units
        multiband: Fit all bands simultaneously. Do not use with `apply` if grouping includes `fid`
        deattenuate: Applies dust and redshift deattenuation (requires fields `mwebv` and `z_final` in `df`)
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
            func = globals()[f"_mag2flux_{sid.lower()}"]

            target[mask] = func(mag[mask])
            error[mask] = func(mag[mask] - e_mag[mask]) - target[mask]

    if deattenuate:
        for sid in df["sid"].unique():
            mask = df["sid"] == sid
            func = globals()[f"_deattenuate_{sid.lower()}"]

            mwebv = df["mwebv"][mask].median()
            zhost = df["z_final"][mask].median()

            func(target[mask], error[mask], band[mask], mwebv, zhost)

    func = _get_fitter(multiband)
    commons = (version, mjd, target, error)

    return func(*commons, band, **kwargs) if multiband else func(*commons, **kwargs)
