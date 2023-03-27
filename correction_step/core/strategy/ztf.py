import warnings

import numpy as np
import pandas as pd

DISTANCE_THRESHOLD = 1.4
SCORE_THRESHOLD = 0.4
CHINR_THRESHOLD = 2
SHARPNR_MAX = 0.1
SHARPNR_MIN = -0.13


def is_corrected(detections: pd.DataFrame) -> pd.Series:
    return detections["distnr"] < DISTANCE_THRESHOLD


def is_dubious(detections: pd.DataFrame) -> pd.Series:
    negative = detections["isdiffpos"] == -1
    corrected = is_corrected(detections)
    first = is_first_corrected(detections)
    return (~corrected & negative) | (first & ~corrected) | (~first & corrected)


def is_stellar(detections: pd.DataFrame) -> pd.Series:
    near_ps1 = detections["distpsnr1"] < DISTANCE_THRESHOLD
    stellar_ps1 = detections["sgscore1"] > SCORE_THRESHOLD

    near_ztf = is_corrected(detections)
    sharpnr_in_range = (SHARPNR_MIN < detections["sharpnr"]) < SHARPNR_MAX
    stellar_ztf = (detections["chinr"] < CHINR_THRESHOLD) & sharpnr_in_range
    return (near_ztf & near_ps1 & stellar_ps1) | (near_ztf & ~near_ps1 & stellar_ztf)


def correct(detections: pd.DataFrame) -> pd.DataFrame:
    aux1 = 10 ** (-.4 * detections["magnr"].astype(float))
    aux2 = 10 ** (-.4 * detections["mag"])
    aux3 = np.maximum(aux1 + detections["isdiffpos"] * aux2, 0.0)
    with warnings.catch_warnings():
        # possible log10 of 0; expected and proper result are given
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mag_corr = -2.5 * np.log10(aux3)

    aux4 = (aux2 * detections["e_mag"]) ** 2 - (aux1 * detections["sigmagnr"].astype(float)) ** 2
    with warnings.catch_warnings():
        # possible sqrt of negative and division by 0; expected and proper results are given
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        e_mag_corr = np.where(aux4 < 0, np.inf, np.sqrt(aux4) / aux3)
        e_mag_corr_ext = aux2 * detections["e_mag"] / aux3

    return pd.DataFrame({"mag_corr": mag_corr, "e_mag_corr": e_mag_corr, "e_mag_corr_ext": e_mag_corr_ext})


def is_first_corrected(detections: pd.DataFrame) -> pd.Series:
    """Whether the first detection for each AID and FID has a nearby known source"""
    corrected = is_corrected(detections)
    idxmin = detections.groupby(["aid", "fid"])["mjd"].transform("idxmin")
    return corrected[idxmin].set_axis(idxmin.index)
