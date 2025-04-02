"""
Transforms that can be applied to `DataFrames` to generate new columns based
on existing columns.
"""

import math
from typing import Callable

import numpy as np
import pandas as pd

ERRORS = {
    1: 0.065,
    2: 0.085,
    3: 0.01,
}

ZERO_MAG = 100.0


# WARN: Use a more appropiate conversion
def _str_to_int(obj: str) -> int:
    """
    Converts a `str` to an `int`.

    Used to transform `oid` into an numeric unique identifier.
    """
    obj_bytes = obj.encode("utf-8")
    return int.from_bytes(obj_bytes, byteorder="big") % 2**30


def objectId_to_oid(df: pd.DataFrame):
    """
    Computes a unique numeric oid based on objectId.

    Takes a `DataFrame` containing the columns:
        - `objectId`
    and uses them to calculate the new columns:
        - `oid`
    """
    df["oid"] = df["objectId"].apply(_str_to_int)


def add_candid(df: pd.DataFrame):
    """
    Computes a candid by adding the objectId and the pid.

    Takes a `DataFrame` containing the columns:
        - `objectId`
        - `pid`
    and uses them to calculate the new columns:
        - `candid`
    """

    # WARN: Use a more appropiate conversion
    df["candid"] = df.apply(
        lambda x: _str_to_int(x["objectId"] + str(x["pid"])), axis=1
    )


def candid_to_measurment_id(df: pd.DataFrame):
    """
    Adds a measurment_id equal to the candid.

    Takes a `DataFrame` containing the columns:
        - `candid`
    and uses them to calculate the new columns:
        - `measurement_id`
    """
    df["measurement_id"] = df["candid"]


def add_tid(df: pd.DataFrame):
    """Adds a tid of zeroes."""
    df["tid"] = 0


def add_sid(df: pd.DataFrame):
    """Adds a sid of zeroes."""
    df["sid"] = 0


def fid_to_band(df: pd.DataFrame):
    """
    Computes the band str from the fid number.

    Takes a `DataFrame` containing the columns:
        - `fid`
    and uses them to calculate the new columns:
        - `band`
    """
    df["band"] = df["fid"]


def jd_to_mjd(df: pd.DataFrame):
    """
    Takes a `DataFrame` containing the columns:
        - `jd`
    and uses them to calculate the new columns:
        - `mjd`
    """
    df["mjd"] = df["jd"] - 2400000.5


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
    df["e_ra"] = df.apply(
        lambda x: x["sigmara"]
        if "sigmara" in x
        else ERRORS[x["fid"]] / abs(math.cos(math.radians(x["dec"]))),
        axis=1,
    )


def sigmadec_to_e_dec(df: pd.DataFrame):
    """
    If there is a sigmadec renames it to e_ra, otherwise it uses the fid errores
    pre defined.

    Takes a `DataFrame` containing the columns:
        - `sigmadec`
        - `fid`
    and uses them to calculate the new columns:
        - `e_dec`
    """
    df["e_dec"] = df.apply(
        lambda x: x["sigmadec"] if "sigmadec" in x else ERRORS[x["fid"]], axis=1
    )


def add_zero_e_ra(df: pd.DataFrame):
    """Adds a e_ra of zeroes."""
    df["e_ra"] = 0


def add_zero_e_dec(df: pd.DataFrame):
    """Adds a e_dec of zeroes."""
    df["e_dec"] = 0


def isdiffpos_to_int(df: pd.DataFrame):
    """
    Converts isdiffpos to a int representation (1 or -1) instead of a string
    ('t', 'f', '1' or '-1').
    """
    df["isdiffpos"] = df["isdiffpos"].apply(lambda x: 1 if x in ["t", "1"] else -1)


def create_add_froced(forced: bool):
    def add_forced(df: pd.DataFrame):
        """Addes froced equal to true"""
        df["forced"] = forced

    return add_forced


def magpsf_to_mag(df: pd.DataFrame):
    df["mag"] = df["magpsf"]


def sigmapsf_to_e_mag(df: pd.DataFrame):
    df["e_mag"] = df["sigmapsf"]


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


def add_drb(df: pd.DataFrame):
    """Adds a drb of None"""
    df["drb"] = None


def add_drbversion(df: pd.DataFrame):
    """Adds a drbversion of None"""
    df["drbversion"] = None


def add_rfid(df: pd.DataFrame):
    """Adds a rfid of None"""
    df["rfid"] = None


def calculate_isdiffpos(df: pd.DataFrame):
    """
    Takes a `DataFrame` containing the columns:
        - `forcediffimflux`
    and uses them to calculate the new columns:
        - `isdiffpos`
    """
    df["isdiffpos"] = np.choose(df["forcediffimflux"] > 0, [-1, 1])


def apply_transforms(
    df: pd.DataFrame, transforms: list[Callable[[pd.DataFrame], None]]
):
    """Applies a list of transforms to the given dataframe"""
    for transform in transforms:
        transform(df)
