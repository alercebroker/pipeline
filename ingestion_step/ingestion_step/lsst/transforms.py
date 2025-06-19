from typing import Literal

import pandas as pd
from db_plugins.db.sql._connection import PsqlDatabase
from idmapper.mapper import catalog_oid_to_masterid

from ingestion_step.core.utils import (
    Transform,
    add_constant_column,
    copy_column,
    rename_column,
)

DIA_SID = 1
SS_SID = 2

band_map = {"g": 1, "r": 2}


def add_oid_to_object(driver: PsqlDatabase, objectType: Literal["dia", "ss"]):
    def _add_oid_to_object(df: pd.DataFrame):
        with driver.session() as session:
            with session.connection() as conn:
                with conn.connection.cursor() as cursor:
                    df["oid"] = df[f"{objectType}ObjectId"].apply(
                        lambda x: int(
                            catalog_oid_to_masterid(
                                "LSST", x, db_cursor=cursor
                            )
                        ),
                    )
                conn.commit()

    return _add_oid_to_object


def add_oid_to_source(driver: PsqlDatabase):
    def _add_oid_to_source(df: pd.DataFrame):
        df["oid"] = df["diaObjectId"]
        if "ssObjectId" in df.columns:
            mask = df["ssObjectId"].notnull()
            df.loc[mask, "oid"] = df["ssObjectId"]

        with driver.session() as session:
            with session.connection() as conn:
                with conn.connection.cursor() as cursor:
                    df["oid"] = df["oid"].apply(
                        lambda x: int(
                            catalog_oid_to_masterid(
                                "LSST", x, db_cursor=cursor
                            )
                        ),
                    )
                conn.commit()

    return _add_oid_to_source


def mjdtai_to_mjd(df: pd.DataFrame):
    df["mjd"] = df["midpointMjdTai"]


def band_to_int(df: pd.DataFrame):
    df.rename(columns={"band": "_band"}, inplace=True)
    df["band"] = df["_band"].map(band_map).astype(pd.Int32Dtype())
    df.drop(columns=["_band"])


def get_source_transforms(driver: PsqlDatabase) -> list[Transform]:
    return [
        add_oid_to_source(driver),
        copy_column("diaSourceId", "measurement_id"),
        mjdtai_to_mjd,
        band_to_int,
    ]


def get_forced_source_transforms(driver: PsqlDatabase) -> list[Transform]:
    return [
        add_oid_to_source(driver),
        copy_column("diaForcedSourceId", "measurement_id"),
        mjdtai_to_mjd,
        band_to_int,
    ]


def get_non_detection_transforms(driver: PsqlDatabase) -> list[Transform]:
    return [add_oid_to_source(driver), mjdtai_to_mjd, band_to_int]


def get_dia_object_transforms(driver: PsqlDatabase) -> list[Transform]:
    return [
        add_oid_to_object(driver, "dia"),
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


def get_ss_object_transforms(driver: PsqlDatabase) -> list[Transform]:
    return [
        add_oid_to_object(driver, "ss"),
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
