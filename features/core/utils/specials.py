import functools
import numpy as np

import mhps
import pandas as pd
from turbofats import FeatureSpace

from .extras.models import fit_sn_model


@functools.lru_cache
def _get_feature_space(features: tuple[str, ...]) -> FeatureSpace:
    return FeatureSpace(features)


def _mag2flux_ztf(mag: np.ndarray) -> np.ndarray:
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)


def mhps4apply(df: pd.DataFrame, t1: float, t2: float) -> pd.Series:
    mag, e_mag, mjd = df[["mag_ml", "e_mag_ml", "mjd"]].T.values

    r, l, h, nz, f = mhps.statistics(mag, e_mag, mjd, t1, t2)
    return pd.Series({"MHPS_ratio": r, "MHPS_low": l, "MHPS_high": h, "MHPS_non_zero": nz, "MHPS_PN_flag": f})


def fats4apply(df: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    space = _get_feature_space(features)

    df = df.set_index("id")[["mag_ml", "e_mag_ml", "mjd"]]
    df = df.rename(columns={"mag_ml": "magnitude", "e_mag_ml": "error", "mjd": "time"})
    return space.calculate_features(df).squeeze(axis="index")


def sn4apply_ztf(df: pd.DataFrame) -> pd.Series:
    mag, e_mag, mjd = df[["mag", "e_mag", "mjd"]].T.values
    flux = _mag2flux_ztf(mag)
    e_flux = _mag2flux_ztf(mag - e_mag) - flux

    result = fit_sn_model(mjd.astype(np.float32), flux.astype(np.float32), e_flux)
    return result.rename({c: f"SPM_{c}" for c in result.index})
