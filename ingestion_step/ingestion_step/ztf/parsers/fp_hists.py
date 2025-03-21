import pandas as pd

from ingestion_step.ztf.parsers.transforms import (
    add_candid,
    add_e_mag,
    add_mag,
    add_sid,
    add_tid,
    add_zero_e_dec,
    add_zero_e_ra,
    apply_transforms,
    candid_to_measurment_id,
    fid_to_band,
    jd_to_mjd,
    objectId_to_oid,
)

fp_transforms = [
    objectId_to_oid,
    add_candid,
    candid_to_measurment_id,
    add_tid,
    add_sid,
    fid_to_band,
    jd_to_mjd,
    add_mag,
    add_e_mag,
    add_zero_e_ra,
    add_zero_e_dec,
]


def _parse_fps_from_fp_hists(fp_hist: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "oid",
        "measurement_id",
        "ra",
        "dec",
        "band",
        "mjd",
        "mag",
        "e_mag",
        # "mag_corr",
        # "e_mag_corr",
        # "e_mag_corr_ext",
        # "isdiffpos",
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
    ]

    forced_photometries = fp_hist[cols]

    return forced_photometries


def parse_fp_hists(
    fp_hists: pd.DataFrame,
) -> pd.DataFrame:
    apply_transforms(fp_hists, fp_transforms)

    forced_photometries = _parse_fps_from_fp_hists(fp_hists)

    return forced_photometries
