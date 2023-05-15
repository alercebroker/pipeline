from typing import Callable

import numpy as np
import pandas as pd


def _mag2flux_ztf(mag: np.ndarray) -> np.ndarray:
    """Magnitude to flux conversion for ZTF.

    Args:
        mag: Magnitudes

    Returns:
        np.ndarray: Fluxes
    """
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)


def spm_ztf(df: pd.DataFrame, spm: Callable[[np.ndarray, np.ndarray, np.ndarray], pd.Series]) -> pd.Series:
    """Fits supernova parametric model parameters for ZTF.

    Note that this always uses uncorrected magnitudes.

    Args:
        df: Frame with magnitudes, errors and times. Expects a single object and band at a time
        spm: Function for fitting SPM. Check functions in module `spm` for typical values

    Returns:
        pd.Series: Fitted parameters
    """
    mag, e_mag, mjd = df[["mag", "e_mag", "mjd"]].T.values
    flux = _mag2flux_ztf(mag)
    e_flux = _mag2flux_ztf(mag - e_mag) - flux

    result = spm(mjd.astype(np.float32), flux.astype(np.float32), e_flux)
    return result.rename({c: f"SPM_{c}" for c in result.index})
