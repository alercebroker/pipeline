import functools

import mhps
import pandas as pd
from turbofats import FeatureSpace


@functools.lru_cache
def _get_feature_space(features: tuple[str, ...]) -> FeatureSpace:
    return FeatureSpace(features)


def mhps4apply(df: pd.DataFrame, t1: float, t2: float) -> pd.Series:
    mag, e_mag, mjd = df[["mag_ml", "e_mag_ml", "mjd"]].T.values

    r, l, h, nz, f = mhps.statistics(mag, e_mag, mjd, t1, t2)
    return pd.Series({"MHPS_ratio": r, "MHPS_low": l, "MHPS_high": h, "MHPS_non_zero": nz, "MHPS_PN_flag": f})


def fats4apply(df: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    space = _get_feature_space(features)

    df = df.set_index("aid")[["mag_ml", "e_mag_ml", "mjd"]]
    df.rename(columns={"mag_ml": "magnitude", "e_mag_ml": "error", "mjd": "time"}, inplace=True)
    return space.calculate_features(df).squeeze(axis="index")
