import numpy as np
import pandas as pd

from ingestion_step.ztf.parsers.transforms import (
    add_candid,
    add_sid,
    add_tid,
    add_zero_e_dec,
    add_zero_e_ra,
    apply_transforms,
    calculate_isdiffpos,
    candid_to_measurment_id,
    fid_to_band,
    forcediffimflux_to_e_mag,
    forcediffimflux_to_mag,
    jd_to_mjd,
    objectId_to_oid,
)

FP_TRANSFORMS = [
    objectId_to_oid,
    add_candid,
    candid_to_measurment_id,
    add_tid,
    add_sid,
    fid_to_band,
    jd_to_mjd,
    forcediffimflux_to_mag,
    forcediffimflux_to_e_mag,
    add_zero_e_ra,
    add_zero_e_dec,
    calculate_isdiffpos,
]
"""
List of mappings applied to the 'fp_hists' `DataFrame` before extracting each
subset of columns.
"""


def _parse_fps_from_fp_hists(fp_hist: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "message_id",
        "oid",
        "sid",
        "tid",
        "measurement_id",
        "ra",
        "e_ra",
        "dec",
        "e_dec",
        "band",
        "mjd",
        "mag",
        "e_mag",
        "rfid",
        # "mag_corr",
        # "e_mag_corr",
        # "e_mag_corr_ext",
        "isdiffpos",
        # "corrected",
        # "dubious",
        "parent_candid",
        "has_stamp",
        "field",
        "rcid",
        "sciinpseeing",
        "scibckgnd",
        "scisigpix",
        "magzpsci",
        "magzpsciunc",
        "magzpscirms",
        "clrcoeff",
        "clrcounc",
        "exptime",
        "adpctdif1",
        "adpctdif2",
        "diffmaglim",
        "programid",
        "procstatus",
        "distnr",
        "ranr",
        "decnr",
        "magnr",
        "sigmagnr",
        "chinr",
        "sharpnr",
        "forced",
    ]

    forced_photometries = fp_hist[cols].replace({np.nan: None})

    return forced_photometries


def parse_fp_hists(
    fp_hists: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parses a `DataFrame` of fp_hists into `forced_photometries`.

    Apply a series of mappings to the original `DataFrame` *mutating it in
    place* then extracts a subset of it`s columns to form the new `DataFrames`.
    """
    apply_transforms(fp_hists, FP_TRANSFORMS)

    forced_photometries = _parse_fps_from_fp_hists(fp_hists)

    return forced_photometries
