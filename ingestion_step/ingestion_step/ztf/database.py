import pandas as pd
from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.sql.models import (
    Detection,
    ForcedPhotometry,
    NonDetection,
    Object,
    ZtfDetection,
    ZtfForcedPhotometry,
    ZtfObject,
)

from ingestion_step.core.database import (
    DETECTION_COLUMNS,
    FORCED_DETECTION_COLUMNS,
    OBJECT_COLUMNS,
    db_statement_builder,
)


def insert_objects(connection: PsqlDatabase, objects_df: pd.DataFrame):
    if len(objects_df) == 0:
        return
    # editing the df
    objects_df["meanra"] = objects_df["ra"]
    objects_df["meandec"] = objects_df["dec"]
    objects_df["firstmjd"] = objects_df["mjd"]
    objects_df["lastmjd"] = objects_df["mjd"]
    objects_df["sigmara"] = None
    objects_df["sigmadec"] = None
    objects_df["deltamjd"] = 0
    objects_df["n_det"] = 1
    objects_df["n_forced"] = 1
    objects_df["n_non_det"] = 1
    objects_df["corrected"] = False
    objects_df["stellar"] = False
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

    object_sql_stmt = db_statement_builder(Object, objects_dict)
    object_ztf_sql_stmt = db_statement_builder(ZtfObject, objects_ztf_dict)

    with connection.session() as session:
        session.execute(object_sql_stmt)
        session.execute(object_ztf_sql_stmt)
        session.commit()


def insert_detections(connection: PsqlDatabase, detections_df: pd.DataFrame):
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

    detection_sql_stmt = db_statement_builder(Detection, detections_dict)
    detection_ztf_sql_stmt = db_statement_builder(ZtfDetection, detections_ztf_dict)

    with connection.session() as session:
        session.execute(detection_sql_stmt)
        session.execute(detection_ztf_sql_stmt)
        session.commit()


def insert_forced_photometry(
    connection: PsqlDatabase, forced_photometry_df: pd.DataFrame
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

    forced_photometry_sql_stmt = db_statement_builder(
        ForcedPhotometry, forced_photometry_dict
    )
    forced_photometry_ztf_sql_stmt = db_statement_builder(
        ZtfForcedPhotometry, forced_photometry_ztf_dict
    )

    with connection.session() as session:
        session.execute(forced_photometry_sql_stmt)
        session.execute(forced_photometry_ztf_sql_stmt)
        session.commit()


def insert_non_detections(connection: PsqlDatabase, non_detections_df: pd.DataFrame):
    if len(non_detections_df) == 0:
        return
    non_detections_df = non_detections_df.reset_index()
    non_detections_dict = non_detections_df[
        [
            "oid",
            "band",
            "mjd",
            "diffmaglim",
        ]
    ].to_dict("records")

    non_detection_sql_stmt = db_statement_builder(NonDetection, non_detections_dict)

    with connection.session() as session:
        session.execute(non_detection_sql_stmt)
        session.commit()
