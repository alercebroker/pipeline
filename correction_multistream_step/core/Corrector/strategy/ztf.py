import warnings

import numpy as np
import pandas as pd

DISTANCE_THRESHOLD = 1.4
SCORE_THRESHOLD = 0.4
CHINR_THRESHOLD = 2
SHARPNR_MAX = 0.1
SHARPNR_MIN = -0.13

_ZERO_MAG = 100.0


#! Fill missing columns with NaN when necessary? => TODO Check if this logic isnt wrong

def is_corrected(detections: pd.DataFrame) -> pd.Series:
    """Whether the nearest source is closer than `DISTANCE_THRESHOLD`"""
    is_new_mask = detections["new"]
    new_detections = detections[is_new_mask]
    old_detections = detections[~is_new_mask]

    corrected_new: pd.Series = new_detections["distnr"] < DISTANCE_THRESHOLD

    if len(old_detections) == 0:
        return corrected_new

    corrected_old: pd.Series = old_detections["corrected"]
    corrected = pd.concat([corrected_new, corrected_old]).loc[detections.index]
    return corrected


def is_dubious(detections: pd.DataFrame) -> pd.Series:
    """A correction/non-correction is dubious if,

    * the flux difference is negative and there is no nearby source, or
    * the first detection for its OID and FID has a nearby source, but the detection doesn't, or
    * the first detection for its OID and FID doesn't have a nearby source, but the detection does.
    """
    is_new_mask = detections["new"]
    new_detections = detections[is_new_mask]
    old_detections = detections[~is_new_mask]

    # Process new detections
    negative = new_detections["isdiffpos"] == -1
    corrected = is_corrected(new_detections)
    first = is_first_corrected(new_detections, corrected)
    dubious_new: pd.Series = (~corrected & negative) | (first & ~corrected) | (~first & corrected)

    if len(old_detections) == 0:
        return dubious_new

    dubious_old: pd.Series = old_detections["dubious"]
    dubious = pd.concat([dubious_new, dubious_old]).loc[detections.index]
    return dubious


def is_stellar(detections: pd.DataFrame) -> pd.Series:
    r"""A source is considered likely stellar in one of two cases.

    First,

    * there is a nearby source according to ZTF, and
    * there is a nearby source according to the PS1 catalogue, and
    * the source has a high start-to-galaxy score according to the PS1 catalogue.

    Second,

    * there is a nearby source according to ZTF, and
    * there is NOT a nearby source according to the PS1 catalogue, and
    * the nearest source sharpness parameter is within a range around zero, and
    * the nearest source $\chi$ parameter is low.
    """
    # Adding column nan in case it doesnt exist (TO CORROBORATE THIS LOGIC) and maybe add it in a better way?
    for col in ["distpsnr1", "sgscore1", "sharpnr", "chinr"]:
        if col not in detections:
            detections[col] = np.nan

    #! CHECK THIS SECTION ABOVE
    ########################################################################################################A
    is_new_mask = detections["new"]
    new_detections = detections[is_new_mask]
    old_detections = detections[~is_new_mask]

    near_ps1 = new_detections["distpsnr1"] < DISTANCE_THRESHOLD
    stellar_ps1 = new_detections["sgscore1"] > SCORE_THRESHOLD
    near_ztf = is_corrected(new_detections)
    sharpnr_in_range = (new_detections["sharpnr"] > SHARPNR_MIN) & (new_detections["sharpnr"] < SHARPNR_MAX)
    stellar_ztf = (new_detections["chinr"] < CHINR_THRESHOLD) & sharpnr_in_range
    stellar_new: pd.Series = (near_ztf & near_ps1 & stellar_ps1) | (
        near_ztf & ~near_ps1 & stellar_ztf
    )

    if len(old_detections) == 0:
        return stellar_new

    stellar_new_with_oid = pd.concat([stellar_new, new_detections["oid"]], axis=1)
    stellar_new_with_oid.rename(columns={0: "stellar"}, inplace=True)
    stellar_new_with_oid.drop_duplicates("oid", keep="first", inplace=True)

    old_detections = old_detections.reset_index()[["index", "oid"]]
    stellar_old = old_detections.merge(stellar_new_with_oid, on="oid", how="left")
    stellar_old: pd.Series = stellar_old.set_index("index")["stellar"]

    # return output in the same order
    stellar = pd.concat([stellar_new, stellar_old]).loc[detections.index]
    return stellar


def correct(detections: pd.DataFrame) -> pd.DataFrame:
    """Apply magnitude correction and compute its associated errors. See `README` for details"""

    need_correction_mask = detections["new"]
    detections_that_need_corr = detections[need_correction_mask]
    detections_that_dont_need_corr = detections[~need_correction_mask]

    corr_mag_column_names = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
    if len(detections_that_dont_need_corr) == 0:
        for col in corr_mag_column_names:
            detections_that_dont_need_corr[col] = []

    corrected_mags_db_observations = detections_that_dont_need_corr[corr_mag_column_names]
    aux1 = 10 ** (-0.4 * detections_that_need_corr["magnr"])
    aux2 = 10 ** (-0.4 * detections_that_need_corr["mag"])
    aux3 = np.maximum(aux1 + detections_that_need_corr["isdiffpos"] * aux2, 0.0)
    with warnings.catch_warnings():
        # possible log10 of 0; this is expected and returned inf is correct value
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mag_corr = -2.5 * np.log10(aux3)

    aux4 = (aux2 * detections_that_need_corr["e_mag"]) ** 2 - (
        aux1 * detections_that_need_corr["sigmagnr"]
    ) ** 2
    
    with warnings.catch_warnings():
        # possible sqrt of negative and division by 0; this is expected and returned inf is correct value
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        e_mag_corr = pd.Series(np.sqrt(aux4) / aux3).where(aux4 >= 0, np.inf) # Handle NA condition 
        e_mag_corr_ext = aux2 * detections_that_need_corr["e_mag"] / aux3

    mask = (detections_that_need_corr["mag"].notna() & 
        np.isclose(detections_that_need_corr["mag"].astype(float), _ZERO_MAG)).to_numpy()
    mag_corr[mask] = np.inf
    e_mag_corr[mask] = np.inf
    e_mag_corr_ext[mask] = np.inf
    
    mask = (detections_that_need_corr["e_mag"].notna() &     
        np.isclose(detections_that_need_corr["e_mag"].astype(float), _ZERO_MAG)).to_numpy()
    e_mag_corr[mask] = np.inf
    e_mag_corr_ext[mask] = np.inf

    corrected_mags_new_observations = pd.DataFrame(
        {"mag_corr": mag_corr, "e_mag_corr": e_mag_corr, "e_mag_corr_ext": e_mag_corr_ext}
    )

    corrected_mags = pd.concat(
        [corrected_mags_new_observations, corrected_mags_db_observations], axis=0
    )
    # return things in the same order
    corrected_mags = corrected_mags.loc[detections.index]
    return corrected_mags


def is_first_corrected(detections: pd.DataFrame, corrected: pd.Series) -> pd.Series:
    """Whether the first detection for each OID and FID has a nearby source"""
    idxmin = detections.groupby(["oid", "band"])["mjd"].transform("idxmin")
    return corrected[idxmin].set_axis(idxmin.index)
