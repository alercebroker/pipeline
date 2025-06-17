import pandas as pd
from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.sql.models import (
    Detection,
    ForcedPhotometry,
    LsstDetection,
    LsstDiaObject,
    LsstForcedPhotometry,
    LsstNonDetection,
    Object,
)

from ingestion_step.core.database import (
    DETECTION_COLUMNS,
    FORCED_DETECTION_COLUMNS,
    OBJECT_COLUMNS,
    db_statement_builder,
)


def insert_dia_objects(driver: PsqlDatabase, dia_objects: pd.DataFrame):
    if len(dia_objects) == 0:
        return

    objects_dict = dia_objects[OBJECT_COLUMNS].to_dict("records")
    objects_dia_lsst_dict = dia_objects[["oid"]].to_dict("records")

    objects_sql_stmt = db_statement_builder(Object, objects_dict)
    objects_dia_lsst_sql_stmt = db_statement_builder(
        LsstDiaObject, objects_dia_lsst_dict
    )

    with driver.session() as session:
        session.execute(objects_sql_stmt)
        session.execute(objects_dia_lsst_sql_stmt)


def insert_ss_objects(driver: PsqlDatabase, ss_objects: pd.DataFrame):
    if len(ss_objects) == 0:
        return

    objects_dict = ss_objects[OBJECT_COLUMNS].to_dict("records")
    objects_ss_lsst_dict = ss_objects[["oid"]].to_dict("records")

    objects_sql_stmt = db_statement_builder(Object, objects_dict)
    objects_dia_lsst_sql_stmt = db_statement_builder(
        LsstDiaObject, objects_ss_lsst_dict
    )

    with driver.session() as session:
        session.execute(objects_sql_stmt)
        session.execute(objects_dia_lsst_sql_stmt)


def insert_sources(driver: PsqlDatabase, sources: pd.DataFrame):
    if len(sources) == 0:
        return

    detections_dict = sources[DETECTION_COLUMNS].to_dict("records")
    detections_lsst_dict = sources[
        [
            "parentDiaSourceId",
            "psfFlux",
            "psfFluxErr",
            "psfFlux_flag",
            "psfFlux_flag_edge",
            "psfFlux_flag_noGoodPixels",
        ]
    ].to_dict("records")

    detections_sql_stmt = db_statement_builder(Detection, detections_dict)
    detections_lsst_sql_stmt = db_statement_builder(
        LsstDetection, detections_lsst_dict
    )

    with driver.session() as session:
        session.execute(detections_sql_stmt)
        session.execute(detections_lsst_sql_stmt)
        session.commit()


def insert_forced_sources(driver: PsqlDatabase, forced_sources: pd.DataFrame):
    if len(forced_sources) == 0:
        return

    forced_detections_dict = forced_sources[FORCED_DETECTION_COLUMNS].to_dict(
        "records"
    )
    forced_detections_lsst_dict = forced_sources[
        [
            "visit",
            "detector",
            "psfFlux",
            "psfFluxErr",
        ]
    ].to_dict("records")

    forced_detections_sql_stmt = db_statement_builder(
        ForcedPhotometry, forced_detections_dict
    )
    forced_detections_lsst_sql_stmt = db_statement_builder(
        LsstForcedPhotometry, forced_detections_lsst_dict
    )

    with driver.session() as session:
        session.execute(forced_detections_sql_stmt)
        session.execute(forced_detections_lsst_sql_stmt)
        session.commit()


def insert_non_detections(driver: PsqlDatabase, non_detections: pd.DataFrame):
    if len(non_detections) == 0:
        return

    non_detections_lsst_dict = non_detections[
        [
            "oid",
            "ccdVisitId",
            "band",
            "mjd",
            "diaNoise",
        ]
    ].to_dict("records")

    non_detections_sql_stmt = db_statement_builder(
        LsstNonDetection, non_detections_lsst_dict
    )

    with driver.session() as session:
        session.execute(non_detections_sql_stmt)
        session.commit()
