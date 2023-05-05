import pandas as pd

FACTOR = 10 ** -(3.9 / 2.5)


def is_corrected(detections: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=detections.index)


def is_dubious(detections: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=detections.index)


def is_stellar(detections: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=detections.index)


def correct(detections: pd.DataFrame) -> pd.DataFrame:
    corr = detections["mag"] * FACTOR
    e_corr = detections["e_mag"] * FACTOR
    return pd.DataFrame({"mag_corr": corr, "e_mag_corr": e_corr, "e_mag_corr_ext": e_corr})
