import functools

import pandas as pd
from turbofats import FeatureSpace


@functools.lru_cache
def _get_feature_space(features: tuple[str, ...]) -> FeatureSpace:
    return FeatureSpace(features)


def turbofats(df: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    space = _get_feature_space(features)

    df = df.set_index("id")[["mag_ml", "e_mag_ml", "mjd"]]
    df = df.rename(columns={"mag_ml": "magnitude", "e_mag_ml": "error", "mjd": "time"})
    return space.calculate_features(df)
