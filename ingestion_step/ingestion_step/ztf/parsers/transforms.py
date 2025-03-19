import math
from typing import Callable

import numpy as np
import pandas as pd

FILTER = {
    1: "g",
    2: "r",
    3: "i",
}

ERRORS = {
    1: 0.065,
    2: 0.085,
    3: 0.01,
}

ZERO_MAG = 100.0


# Use a more appropiate conversion
def _str_to_int(obj: str) -> int:
    obj_bytes = obj.encode("utf-8")
    return int.from_bytes(obj_bytes, byteorder="big")


def objectId_to_oid(df: pd.DataFrame):
    df["oid"] = df["objectId"].apply(_str_to_int)


def add_candid(df: pd.DataFrame):
    df["candid"] = df.apply(lambda x: x["objectId"] + str(x["pid"]), axis=1)


def candid_to_measurment_id(df: pd.DataFrame):
    df["measurement_id"] = df["candid"]


def add_tid(df: pd.DataFrame):
    df["tid"] = 0


def add_sid(df: pd.DataFrame):
    df["sid"] = 0


def _to_filter_str(fid: int) -> str:
    return FILTER[fid]


def fid_to_band(df: pd.DataFrame):
    df["band"] = df["fid"].apply(_to_filter_str)


def jd_to_mjd(df: pd.DataFrame):
    df["mjd"] = df["jd"] - 2400000.5


# TODO: Update to use numpy instead
def _e_ra(sigmara: float | None, dec: float, fid: int) -> float:
    if sigmara:
        return sigmara
    try:
        return ERRORS[fid] / abs(math.cos(math.radians(dec)))
    except ZeroDivisionError:
        return float("nan")


def sigmara_to_e_ra(df: pd.DataFrame):
    df["e_ra"] = df.apply(lambda x: _e_ra(x["sigmara"], x["dec"], x["fid"]), axis=1)


def sigmadec_to_e_dec(df: pd.DataFrame):
    df["e_dec"] = df.apply(
        lambda x: x["sigmadec"] if x["sigmadec"] else ERRORS[x["fid"]], axis=1
    )


def add_zero_e_ra(df: pd.DataFrame):
    df["e_ra"] = 0


def add_zero_e_dec(df: pd.DataFrame):
    df["e_dec"] = 0


def isdiffpos_to_int(df: pd.DataFrame):
    df["isdiffpos"] = df["isdiffpos"].apply(lambda x: 1 if x in ["t", "1"] else -1)


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


def add_mag(df: pd.DataFrame):
    df["mag"] = df.apply(
        lambda x: _calculate_mag(
            x["magzpsci"], x["forcediffimflux"], x["forcediffimfluxunc"]
        )[0],
        axis=1,
    )


def add_e_mag(df: pd.DataFrame):
    df["e_mag"] = df.apply(
        lambda x: _calculate_mag(
            x["magzpsci"], x["forcediffimflux"], x["forcediffimfluxunc"]
        )[1],
        axis=1,
    )


def add_drb(df: pd.DataFrame):
    df["drb"] = None


def add_drbversion(df: pd.DataFrame):
    df["drbversion"] = None


def add_rfid(df: pd.DataFrame):
    df["rfid"] = None


def apply_transforms(
    df: pd.DataFrame, transforms: list[Callable[[pd.DataFrame], None]]
):
    for transform in transforms:
        transform(df)
