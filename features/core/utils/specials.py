import functools

import mhps
import pandas as pd
from turbofats import FeatureSpace


@functools.lru_cache
def _get_feature_space(features: tuple[str, ...]) -> FeatureSpace:
    return FeatureSpace(features)


def mhps4apply(df: pd.DataFrame, t1: float, t2: float) -> pd.Series:
    mag, e_mag, mjd = df[["mag_ml", "e_mag_ml", "mjd"]].T.values
    ratio, low, high, nonzero, flag = mhps.statistics(mag, e_mag, mjd, t1, t2)

    return pd.Series(
        {"MHPS_ratio": ratio, "MHPS_low": low, "MHPS_high": high, "MHPS_non_zero": nonzero, "MHPS_PN_flag": flag}
    )


def fats4apply(df: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    space = _get_feature_space(features)

    df = df.set_index("aid")[["mag_ml", "e_mag_ml", "mjd"]]
    df.rename(columns={"mag_ml": "magnitude", "e_mag_ml": "error", "mjd": "time"}, inplace=True)
    return space.calculate_features(df).squeeze(axis="index")
