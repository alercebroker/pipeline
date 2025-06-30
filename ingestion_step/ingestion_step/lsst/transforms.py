import pandas as pd

from ingestion_step.core.utils import (
    Transform,
    add_constant_column,
    copy_column,
    rename_column,
)

DIA_SID = 1
SS_SID = 2

band_map = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}


def dia_object_id_to_oid(df: pd.DataFrame):
    df["oid"] = df["diaObjectId"]


def ss_object_id_to_oid(df: pd.DataFrame):
    df["oid"] = df["ssObjectId"]


def add_oid(df: pd.DataFrame):
    df["oid"] = df["diaObjectId"]
    if "ssObjectId" in df.columns:
        mask = df["ssObjectId"].notnull()
        df.loc[mask, "oid"] = df["ssObjectId"]


def add_sid(df: pd.DataFrame):
    df["sid"] = DIA_SID
    if "ssObjectId" in df.columns:
        mask = df["ssObjectId"].notnull()
        df.loc[mask, "sid"] = SS_SID


def mjdtai_to_mjd(df: pd.DataFrame):
    df["mjd"] = df["midpointMjdTai"]


def band_to_int(df: pd.DataFrame):
    df.rename(columns={"band": "_band"}, inplace=True)
    df["band"] = df["_band"].map(band_map).astype(pd.Int32Dtype())
    df.drop(columns=["_band"])


def get_source_transforms() -> list[Transform]:
    return [
        add_oid,
        add_sid,
        copy_column("diaSourceId", "measurement_id"),
        mjdtai_to_mjd,
        band_to_int,
    ]


def get_forced_source_transforms() -> list[Transform]:
    return [
        add_oid,
        add_sid,
        copy_column("diaForcedSourceId", "measurement_id"),
        mjdtai_to_mjd,
        band_to_int,
    ]


def get_non_detection_transforms() -> list[Transform]:
    return [
        add_oid,
        add_sid,
        mjdtai_to_mjd,
        band_to_int,
    ]


def get_dia_object_transforms() -> list[Transform]:
    return [
        dia_object_id_to_oid,
        add_constant_column("tid", 0, pd.Int32Dtype()),
        add_constant_column("sid", DIA_SID, pd.Int32Dtype()),
        mjdtai_to_mjd,
        copy_column("mjd", "firstmjd"),
        copy_column("mjd", "lastmjd"),
        copy_column("ra", "meanra"),
        copy_column("dec", "meandec"),
    ]


def get_ss_object_transforms() -> list[Transform]:
    return [
        ss_object_id_to_oid,
        add_constant_column("tid", 0, pd.Int32Dtype()),
        add_constant_column("sid", SS_SID, pd.Int32Dtype()),
        mjdtai_to_mjd,
        copy_column("mjd", "firstmjd"),
        copy_column("mjd", "lastmjd"),
        rename_column("ra", "meanra"),
        rename_column("dec", "meandec"),
    ]
