"""
Transforms that can be applied to `DataFrames` to generate new columns based
on existing columns.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from idmapper.mapper import catalog_oid_to_masterid

from ingestion_step.core.utils import add_constant_column, copy_column, deduplicate

ERRORS = {
    1: 0.065,
    2: 0.085,
    3: 0.01,
}

ZERO_MAG = 100.0


def objectId_to_oid(df: pd.DataFrame):
    """
    Computes a unique numeric oid based on objectId.

    Takes a `DataFrame` containing the columns:
        - `objectId`
    and uses them to calculate the new columns:
        - `oid`
    """
    df["oid"] = df["objectId"].apply(lambda x: int(catalog_oid_to_masterid("ZTF", x)))


def jd_to_mjd(df: pd.DataFrame):
    """
    Takes a `DataFrame` containing the columns:
        - `jd`
    and uses them to calculate the new columns:
        - `mjd`
    """
    df["mjd"] = (df["jd"] - 2400000.5).astype("Float64")


# TODO: Update to use numpy vectorized function instead of element wise
def sigmara_to_e_ra(df: pd.DataFrame):
    """
    If there is a sigmara renames it to e_ra, otherwise it gets computes from
    the dec and fid.

    Takes a `DataFrame` containing the columns:
        - `sigmara`
        - `dec`
        - `fid`
    and uses them to calculate the new columns:
        - `e_ra`
    """

    def _sigmara_to_e_ra(x: dict[str, Any]) -> float:
        if "sigmara" in x:
            return x["sigmara"]
        if x["dec"] is not pd.NA and x["dec"] is not None:
            return ERRORS[x["fid"]] / abs(math.cos(math.radians(x["dec"])))
        else:
            return float("nan")

    df["e_ra"] = df.apply(
        _sigmara_to_e_ra,
        axis=1,
    )


def sigmadec_to_e_dec(df: pd.DataFrame):
    """
    If there is a sigmadec renames it to e_ra, otherwise it uses the fid errors
    pre-defined.

    Takes a `DataFrame` containing the columns:
        - `sigmadec`
        - `fid`
    and uses them to calculate the new columns:
        - `e_dec`
    """
    df["e_dec"] = df.apply(
        lambda x: x["sigmadec"] if "sigmadec" in x else ERRORS[x["fid"]],
        axis=1,
    )


def isdiffpos_to_int(df: pd.DataFrame):
    """
    Converts isdiffpos to an int representation (1 or -1) instead of a string
    ('t', 'f', '1' or '-1').
    """
    df["isdiffpos"] = df["isdiffpos"].apply(
        lambda x: 1 if pd.notna(x) and x in ["t", "1"] else -1
    )


# TODO: Check that the results are correct and add a small explanation.
# Currently this function is a copy of the one implemented in prv_candidates_step
def _calculate_mag(
    magzpsci: float, forcediffimflux: float, forcediffimfluxunc: float
) -> tuple[float, float]:
    if np.isclose(forcediffimflux, -99999):
        return ZERO_MAG, ZERO_MAG

    flux2uJy = 10.0 ** ((8.9 - magzpsci) / 2.5) * 1.0e6

    mag = -2.5 * np.log10(np.abs(forcediffimflux * flux2uJy)) + 23.9

    if np.isclose(forcediffimfluxunc, -99999):
        e_mag = ZERO_MAG
    else:
        e_mag = (
            1.0857 * forcediffimfluxunc * flux2uJy / np.abs(forcediffimflux * flux2uJy)
        )

    return mag, e_mag


# TODO: Update to use numpy vectorized function instead of element wise
def forcediffimflux_to_mag(df: pd.DataFrame):
    """
    Takes a `DataFrame` containing the columns:
        - `magzpsci`
        - `forcediffimflux`
        - `forcediffimfluxunc`
    and uses them to calculate the new columns:
        - `mag`
    """
    df["mag"] = df.apply(
        lambda x: _calculate_mag(
            x["magzpsci"], x["forcediffimflux"], x["forcediffimfluxunc"]
        )[0],
        axis=1,
    )


def forcediffimflux_to_e_mag(df: pd.DataFrame):
    """
    Takes a `DataFrame` containing the columns:
        - `magzpsci`
        - `forcediffimflux`
        - `forcediffimfluxunc`
    and uses them to calculate the new columns:
        - `e_mag`
    """
    df["e_mag"] = df.apply(
        lambda x: _calculate_mag(
            x["magzpsci"], x["forcediffimflux"], x["forcediffimfluxunc"]
        )[1],
        axis=1,
    )


def calculate_isdiffpos(df: pd.DataFrame):
    """
    Takes a `DataFrame` containing the columns:
        - `forcediffimflux`
    and uses them to calculate the new columns:
        - `isdiffpos`
    """
    df["isdiffpos"] = np.choose(df["forcediffimflux"] > 0, [-1, 1])


def filter_by_forcediffimflux(df: pd.DataFrame):
    df.drop(
        df[
            (df["forcediffimflux"].isna())
            | (df["forcediffimflux"] == 0)
            | (np.isclose(df["forcediffimflux"], -99999))
        ].index,
        inplace=True,
    )
    df.drop(
        df[
            (df["forcediffimfluxunc"].isna())
            | (df["forcediffimfluxunc"] == 0)
            | (np.isclose(df["forcediffimfluxunc"], -99999))
        ].index,
        inplace=True,
    )
    df.reset_index(inplace=True, drop=True)


CANDIDATES_TRANSFORMS = [
    objectId_to_oid,
    copy_column("candid", "measurement_id"),
    add_constant_column("tid", 0, pd.Int32Dtype()),
    add_constant_column("sid", 0, pd.Int32Dtype()),
    isdiffpos_to_int,
    jd_to_mjd,
    deduplicate(["oid", "measurement_id"], sort="mjd"),
    sigmara_to_e_ra,
    sigmadec_to_e_dec,
    copy_column("fid", "band"),
    copy_column("magpsf", "mag"),
    copy_column("sigmapsf", "e_mag"),
]

PRV_CANDIDATES_TRANSFORMS = [
    objectId_to_oid,
    copy_column("candid", "measurement_id"),
    add_constant_column("tid", 0, pd.Int32Dtype()),
    add_constant_column("sid", 0, pd.Int32Dtype()),
    isdiffpos_to_int,
    jd_to_mjd,
    sigmara_to_e_ra,
    sigmadec_to_e_dec,
    copy_column("fid", "band"),
    copy_column("magpsf", "mag"),
    copy_column("sigmapsf", "e_mag"),
    add_constant_column("drb", None, pd.Float32Dtype()),
    add_constant_column("drbversion", None, pd.StringDtype()),
    add_constant_column("rfid", None, pd.Int64Dtype()),
]

FP_TRANSFORMS = [
    add_constant_column("tid", 0, pd.Int32Dtype()),
    add_constant_column("sid", 0, pd.Int32Dtype()),
    add_constant_column("e_ra", 0.0, pd.Float64Dtype()),
    add_constant_column("e_dec", 0.0, pd.Float64Dtype()),
    objectId_to_oid,
    copy_column("pid", "candid"),
    copy_column("candid", "measurement_id"),
    jd_to_mjd,
    forcediffimflux_to_mag,
    forcediffimflux_to_e_mag,
    copy_column("fid", "band"),
    calculate_isdiffpos,
    filter_by_forcediffimflux,
]
