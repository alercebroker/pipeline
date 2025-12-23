from typing import Literal

import pandas as pd

from ingestion_step.core.utils import (
    Transform,
    add_constant_column,
    copy_column,
)

DIA_SID = 1
SS_SID = 2
TID = 1

band_map = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}


def add_oid(df: pd.DataFrame):
    added_tmp_column = False
    if "ssObjectId" not in df.columns:
        df["ssObjectId"] = pd.NA
        added_tmp_column = True

    df["oid"] = df["ssObjectId"]
    mask = df["diaObjectId"].notnull() & (df["diaObjectId"] != 0)
    df.loc[mask, "oid"] = df["diaObjectId"]

    if added_tmp_column:
        df.drop(columns=["ssObjectId"], inplace=True)


def add_sid_to_source(df: pd.DataFrame):
    added_tmp_column = False
    if "ssObjectId" not in df.columns:
        df["ssObjectId"] = pd.NA

        added_tmp_column = True

    df["sid"] = SS_SID
    if "diaObjectId" in df.columns:
        mask = df["diaObjectId"].notnull() & (df["diaObjectId"] != 0)
        df.loc[mask, "sid"] = DIA_SID

    if added_tmp_column:
        df.drop(columns=["ssObjectId"], inplace=True)


def band_to_int(df: pd.DataFrame):
    df.rename(columns={"band": "_band"}, inplace=True)
    df["band"] = df["_band"].map(band_map).astype(pd.Int32Dtype())
    df.drop(columns=["_band"], inplace=True)


def deduplicate(columns: list[str], sort: str | list[str] | None = None):
    def _deduplicate(df: pd.DataFrame):
        if sort is not None:
            df.sort_values(sort, inplace=True)
        df.drop_duplicates(subset=columns, keep="first", inplace=True)

    return _deduplicate


def drop_na(columns: list[str], axis: Literal[0, 1, "index", "columns"] = 0):
    def _drop_na(df: pd.DataFrame):
        df.dropna(axis=axis, subset=columns, how="any")

    return _drop_na


def get_source_transforms() -> list[Transform]:
    return [
        add_oid,
        add_sid_to_source,
        copy_column("diaSourceId", "measurement_id"),
        deduplicate(["oid", "sid", "measurement_id"], sort="midpointMjdTai"),
        copy_column("midpointMjdTai", "mjd"),
        band_to_int,
    ]


def get_forced_source_transforms() -> list[Transform]:
    return [
        add_oid,
        add_sid_to_source,
        copy_column("diaForcedSourceId", "measurement_id"),
        deduplicate(["oid", "sid", "measurement_id"], sort="midpointMjdTai"),
        copy_column("midpointMjdTai", "mjd"),
        band_to_int,
    ]


def get_ss_source_transforms() -> list[Transform]:
    return [
        copy_column("ssObjectId", "oid"),
        copy_column("diaSourceId", "measurement_id"),
        deduplicate(["measurement_id"]),
    ]


def get_dia_object_transforms() -> list[Transform]:
    return [
        copy_column("diaObjectId", "oid"),
        add_constant_column("tid", TID, pd.Int32Dtype()),
        add_constant_column("sid", DIA_SID, pd.Int32Dtype()),
        deduplicate(["oid", "sid"], sort="midpointMjdTai"),
        copy_column("midpointMjdTai", "mjd"),
        copy_column("mjd", "firstmjd"),
        copy_column("mjd", "lastmjd"),
        copy_column("ra", "meanra"),
        copy_column("dec", "meandec"),
    ]


def get_mpc_orbits_transforms() -> list[Transform]:
    return [
        drop_na(["ssObjectId"]),
        deduplicate(["ssObjectId"], sort="midpointMjdTai"),
        copy_column("midpointMjdTai", "mjd"),
    ]
