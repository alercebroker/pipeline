import pandas as pd

from ingestion_step.core.utils import add_constant_column, copy_column, rename_column

DIA_SID = 1
SS_SID = 2

band_map = {"g": 1, "r": 2}


def dia_object_id_to_oid(df: pd.DataFrame):
    # TODO: Will fail in prod, must replace for a better mapping
    df["oid"] = df["diaObjectId"]


def ss_object_id_to_oid(df: pd.DataFrame):
    # TODO: Will fail in prod, must replace for a better mapping
    df["oid"] = df["ssObjectId"]


def add_oid(df: pd.DataFrame):
    # TODO: Will fail in prod, must replace for a better mapping
    df["oid"] = df["diaObjectId"]
    if "ssObjectId" in df.columns:
        mask = df["ssObjectId"].notnull()
        df.loc[mask, "oid"] = df["ssObjectId"]


def mjdtai_to_mjd(df: pd.DataFrame):
    # TODO: Will fail in prod, must replace for a better mapping
    df["mjd"] = df["midpointMjdTai"]


def band_to_int(df: pd.DataFrame):
    df.rename(columns={"band": "_band"}, inplace=True)
    df["band"] = df["_band"].map(band_map).astype(pd.Int32Dtype())
    df.drop(columns=["_band"])


SOURCE_TRANSFORMS = [
    add_oid,
    copy_column("diaSourceId", "measurement_id"),
    mjdtai_to_mjd,
    band_to_int,
]

FORCED_SOURCE_TRANSFORMS = [
    add_oid,
    copy_column("diaForcedSourceId", "measurement_id"),
    mjdtai_to_mjd,
    band_to_int,
]

NON_DETECTION_TRANSFORMS = [add_oid, mjdtai_to_mjd, band_to_int]

DIA_OBJECT_TRANSFORMS = [
    dia_object_id_to_oid,
    add_constant_column("tid", 0, pd.Int32Dtype()),
    add_constant_column("sid", DIA_SID, pd.Int32Dtype()),
    mjdtai_to_mjd,
    copy_column("mjd", "firstmjd"),
    copy_column("mjd", "lastmjd"),
    copy_column("ra", "meanra"),
    copy_column("dec", "meandec"),
    add_constant_column("sigmara", None, pd.Float64Dtype()),
    add_constant_column("sigmadec", None, pd.Float64Dtype()),
    add_constant_column("deltamjd", 0.0, pd.Float64Dtype()),
    add_constant_column("n_det", 1, pd.Int32Dtype()),
    add_constant_column("n_forced", 1, pd.Int32Dtype()),
    add_constant_column("n_non_det", 1, pd.Int32Dtype()),
    add_constant_column("corrected", False, pd.BooleanDtype()),
    add_constant_column("stellar", False, pd.BooleanDtype()),
]
SS_OBJECT_TRANSFORMS = [
    ss_object_id_to_oid,
    add_constant_column("tid", 0, pd.Int32Dtype()),
    add_constant_column("sid", SS_SID, pd.Int32Dtype()),
    mjdtai_to_mjd,
    copy_column("mjd", "firstmjd"),
    copy_column("mjd", "lastmjd"),
    rename_column("ra", "meanra"),
    rename_column("dec", "meandec"),
    add_constant_column("sigmara", None, pd.Float64Dtype()),
    add_constant_column("sigmadec", None, pd.Float64Dtype()),
    add_constant_column("deltamjd", 0.0, pd.Float64Dtype()),
    add_constant_column("n_det", 1, pd.Int32Dtype()),
    add_constant_column("n_forced", 1, pd.Int32Dtype()),
    add_constant_column("n_non_det", 1, pd.Int32Dtype()),
    add_constant_column("corrected", False, pd.BooleanDtype()),
    add_constant_column("stellar", False, pd.BooleanDtype()),
]
