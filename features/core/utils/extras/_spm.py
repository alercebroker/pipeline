from typing import Callable

import numpy as np
import pandas as pd


def _mag2flux_ztf(mag: np.ndarray) -> np.ndarray:
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)


def spm_ztf(df: pd.DataFrame, spm: Callable[[np.ndarray, np.ndarray, np.ndarray], pd.Series]) -> pd.Series:
    mag, e_mag, mjd = df[["mag", "e_mag", "mjd"]].T.values
    flux = _mag2flux_ztf(mag)
    e_flux = _mag2flux_ztf(mag - e_mag) - flux

    result = spm(mjd.astype(np.float32), flux.astype(np.float32), e_flux)
    return result.rename({c: f"SPM_{c}" for c in result.index})
