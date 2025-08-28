import pandas as pd
from db_plugins.db.sql._connection import AsyncPsqlDatabase
from db_plugins.db.sql.models import (
    Detection,
    ForcedPhotometry,
    Object,
    ZtfDetection,
    ZtfForcedPhotometry,
    ZtfNonDetection,
    ZtfObject,
)

from ingestion_step.core.database import (
    DETECTION_COLUMNS,
    FORCED_DETECTION_COLUMNS,
    OBJECT_COLUMNS,
    db_statement_builder,
)

CHUNK_SIZE = 1000


async def insert_objects(driver: AsyncPsqlDatabase, objects_df: pd.DataFrame):
    if len(objects_df) == 0:
        return
    # editing the df
    objects_df["meanra"] = objects_df["ra"]
    objects_df["meandec"] = objects_df["dec"]
    objects_df["firstmjd"] = objects_df["mjd"]
    objects_df["lastmjd"] = objects_df["mjd"]
    objects_df["g_r_max"] = None
    objects_df["g_r_max_corr"] = None
    objects_df["g_r_mean"] = None
    objects_df["g_r_mean_corr"] = None

    objects_df = objects_df.reset_index()

    objects_df_parsed = objects_df[OBJECT_COLUMNS]
    objects_dict = objects_df_parsed.to_dict("records")
    objects_ztf_df_parsed = objects_df[
        [
            "oid",
            "g_r_max",
            "g_r_max_corr",
            "g_r_mean",
            "g_r_mean_corr",
        ]
    ]
    objects_ztf_dict = objects_ztf_df_parsed.to_dict("records")

    async with driver.session() as session:
        for i in range(0, len(objects_dict), CHUNK_SIZE):
            object_sql_stmt = db_statement_builder(
                Object, objects_dict[i : i + CHUNK_SIZE]
            )
            object_ztf_sql_stmt = db_statement_builder(
                ZtfObject, objects_ztf_dict[i : i + CHUNK_SIZE]
            )
            await session.execute(object_sql_stmt)
            await session.execute(object_ztf_sql_stmt)
        await session.commit()


async def insert_detections(driver: AsyncPsqlDatabase, detections_df: pd.DataFrame):
    if len(detections_df) == 0:
        return
    # editing the df
    detections_df["magpsf_corr"] = None
    detections_df["sigmapsf_corr"] = None
    detections_df["sigmapsf_corr_ext"] = None
    detections_df["corrected"] = False
    detections_df["dubious"] = False

    detections_df = detections_df.reset_index()

    detections_df_parsed = detections_df[DETECTION_COLUMNS]
    detections_dict = detections_df_parsed.to_dict("records")
    detections_ztf_df_parsed = detections_df[
        [
            "oid",
            "sid",
            "measurement_id",
            "pid",
            "diffmaglim",
            "isdiffpos",
            "nid",
            "magpsf",
            "sigmapsf",
            "magap",
            "sigmagap",
            "distnr",
            "rb",
            "rbversion",
            "drb",
            "drbversion",
            "magapbig",
            "sigmagapbig",
            "rfid",
            "magpsf_corr",
            "sigmapsf_corr",
            "sigmapsf_corr_ext",
            "corrected",
            "dubious",
            "parent_candid",
            "has_stamp",
        ]
    ]

    detections_ztf_dict = detections_ztf_df_parsed.to_dict("records")

    async with driver.session() as session:
        for i in range(0, len(detections_dict), CHUNK_SIZE):
            detection_sql_stmt = db_statement_builder(
                Detection, detections_dict[i : i + CHUNK_SIZE]
            )
            detection_ztf_sql_stmt = db_statement_builder(
                ZtfDetection, detections_ztf_dict[i : i + CHUNK_SIZE]
            )
            await session.execute(detection_sql_stmt)
            await session.execute(detection_ztf_sql_stmt)
        await session.commit()


async def insert_forced_photometry(
    driver: AsyncPsqlDatabase, forced_photometry_df: pd.DataFrame
):
    if len(forced_photometry_df) == 0:
        return
    # editing the df
    forced_photometry_df["mag_corr"] = None
    forced_photometry_df["e_mag_corr"] = None
    forced_photometry_df["e_mag_corr_ext"] = None
    forced_photometry_df["corrected"] = False
    forced_photometry_df["dubious"] = False

    forced_photometry_df = forced_photometry_df.reset_index()

    forced_photometry_df_parsed = forced_photometry_df[FORCED_DETECTION_COLUMNS]
    forced_photometry_dict = forced_photometry_df_parsed.to_dict("records")
    forced_photometry_ztf_df_parsed = forced_photometry_df[
        [
            "oid",
            "sid",
            "measurement_id",
            "pid",
            "mag",
            "e_mag",
            "mag_corr",
            "e_mag_corr",
            "e_mag_corr_ext",
            "isdiffpos",
            "corrected",
            "dubious",
            "parent_candid",
            "has_stamp",
            "field",
            "rcid",
            "rfid",
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
    ]
    forced_photometry_ztf_dict = forced_photometry_ztf_df_parsed.to_dict("records")

    async with driver.session() as session:
        for i in range(0, len(forced_photometry_dict), CHUNK_SIZE):
            forced_photometry_sql_stmt = db_statement_builder(
                ForcedPhotometry, forced_photometry_dict[i : i + CHUNK_SIZE]
            )
            forced_photometry_ztf_sql_stmt = db_statement_builder(
                ZtfForcedPhotometry, forced_photometry_ztf_dict[i : i + CHUNK_SIZE]
            )
            await session.execute(forced_photometry_sql_stmt)
            await session.execute(forced_photometry_ztf_sql_stmt)
        await session.commit()


async def insert_non_detections(
    driver: AsyncPsqlDatabase, non_detections_df: pd.DataFrame
):
    if len(non_detections_df) == 0:
        return
    non_detections_df = non_detections_df.reset_index()
    non_detections_dict = non_detections_df[
        [
            "oid",
            "sid",
            "band",
            "mjd",
            "diffmaglim",
        ]
    ].to_dict("records")

    async with driver.session() as session:
        for i in range(0, len(non_detections_dict), CHUNK_SIZE):
            non_detection_sql_stmt = db_statement_builder(
                ZtfNonDetection, non_detections_dict[i : i + CHUNK_SIZE]
            )
            await session.execute(non_detection_sql_stmt)
        await session.commit()
