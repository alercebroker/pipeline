import pandas as pd

from ingestion_step.core.utils import (
    Transform,
    add_constant_column,
    copy_column,
    rename_column,
)

DIA_SID = 1
SS_SID = 2
TID = 1

band_map = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}


def add_oid(df: pd.DataFrame):
    added_column = False
    if "ssObjectId" not in df.columns:
        df["ssObjectId"] = pd.NA
        added_column = True

    df["oid"] = df["ssObjectId"]
    mask = df["diaObjectId"].notnull() & (df["diaObjectId"] != 0)
    df.loc[mask, "oid"] = df["diaObjectId"]

    if added_column:
        df.drop(columns=["ssObjectId"], inplace=True)


def add_sid_to_source(df: pd.DataFrame):
    added_column = False
    if "ssObjectId" not in df.columns:
        df["ssObjectId"] = pd.NA

        added_column = True

    df["sid"] = SS_SID
    if "diaObjectId" in df.columns:
        mask = df["diaObjectId"].notnull() & (df["diaObjectId"] != 0)
        df.loc[mask, "sid"] = DIA_SID

    if added_column:
        df.drop(columns=["ssObjectId"], inplace=True)


def band_to_int(df: pd.DataFrame):
    df.rename(columns={"band": "_band"}, inplace=True)
    df["band"] = df["_band"].map(band_map).astype(pd.Int32Dtype())
    df.drop(columns=["_band"], inplace=True)


def deduplicate(columns: list[str]):
    def _deduplicate(df: pd.DataFrame):
        df.drop_duplicates(subset=columns, inplace=True)

    return _deduplicate


def get_source_transforms() -> list[Transform]:
    return [
        add_oid,
        add_sid_to_source,
        copy_column("diaSourceId", "measurement_id"),
        deduplicate(["oid", "sid", "measurement_id"]),
        copy_column("midpointMjdTai", "mjd"),
        band_to_int,
    ]


def get_ss_source_transforms() -> list[Transform]:
    return []


def get_forced_source_transforms() -> list[Transform]:
    return [
        add_oid,
        add_sid_to_source,
        copy_column("diaForcedSourceId", "measurement_id"),
        deduplicate(["oid", "sid", "measurement_id"]),
        copy_column("midpointMjdTai", "mjd"),
        band_to_int,
    ]


def get_non_detection_transforms() -> list[Transform]:
    return [
        add_oid,
        add_sid_to_source,
        copy_column("midpointMjdTai", "mjd"),
        band_to_int,
    ]


def get_dia_object_transforms() -> list[Transform]:
    return [
        copy_column("diaObjectId", "oid"),
        add_constant_column("tid", TID, pd.Int32Dtype()),
        add_constant_column("sid", DIA_SID, pd.Int32Dtype()),
        deduplicate(["oid", "sid"]),
        copy_column("midpointMjdTai", "mjd"),
        copy_column("mjd", "firstmjd"),
        copy_column("mjd", "lastmjd"),
        copy_column("ra", "meanra"),
        copy_column("dec", "meandec"),
    ]


def get_ss_object_transforms() -> list[Transform]:
    return [
        copy_column("ssObjectId", "oid"),
        add_constant_column("tid", TID, pd.Int32Dtype()),
        add_constant_column("sid", SS_SID, pd.Int32Dtype()),
        copy_column("midpointMjdTai", "mjd"),
        copy_column("mjd", "firstmjd"),
        copy_column("mjd", "lastmjd"),
        rename_column("ra", "meanra"),
        rename_column("dec", "meandec"),
    ]


def get_mpcorb_transforms() -> list[Transform]:
    return []
