import pandas as pd
from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.sql.models import (
    Detection,
    ForcedPhotometry,
    LsstDetection,
    LsstDiaObject,
    LsstForcedPhotometry,
    LsstNonDetection,
    LsstSsObject,
    Object,
)

from ingestion_step.core.database import (
    DETECTION_COLUMNS,
    FORCED_DETECTION_COLUMNS,
    OBJECT_COLUMNS,
    db_statement_builder,
)

from sqlalchemy.dialects.postgresql import insert as pg_insert


def bulk_insert_on_conflict_do_nothing(session, model, records, conflict_columns=None):
    if not records:
        return
    stmt = pg_insert(model).values(records)
    if conflict_columns:
        stmt = stmt.on_conflict_do_nothing(index_elements=conflict_columns)
    else:
        # Let SQLAlchemy/DB use the primary key
        stmt = stmt.on_conflict_do_nothing()
    session.execute(stmt)


def insert_dia_objects(session, dia_objects: pd.DataFrame):
    if dia_objects.empty:
        return

    objects_dict = dia_objects[OBJECT_COLUMNS].to_dict("records")
    objects_dia_lsst_dict = dia_objects[["oid"]].to_dict("records")

    bulk_insert_on_conflict_do_nothing(session, Object, objects_dict)
    bulk_insert_on_conflict_do_nothing(session, LsstDiaObject, objects_dia_lsst_dict)


def insert_ss_objects(session, ss_objects: pd.DataFrame):
    if ss_objects.empty:
        return

    objects_dict = ss_objects[OBJECT_COLUMNS].to_dict("records")
    objects_ss_lsst_dict = ss_objects[["oid"]].to_dict("records")

    bulk_insert_on_conflict_do_nothing(session, Object, objects_dict)
    bulk_insert_on_conflict_do_nothing(session, LsstSsObject, objects_ss_lsst_dict)


def insert_sources(session, sources: pd.DataFrame):
    if sources.empty:
        return

    detections_dict = sources[DETECTION_COLUMNS].to_dict("records")
    detections_lsst_dict = sources[
        [
            "oid",
            "sid",
            "measurement_id",
            "parentDiaSourceId",
            "psfFlux",
            "psfFluxErr",
            "psfFlux_flag",
            "psfFlux_flag_edge",
            "psfFlux_flag_noGoodPixels",
            "raErr",
            "decErr",
        ]
    ].to_dict("records")

    bulk_insert_on_conflict_do_nothing(session, Detection, detections_dict)
    bulk_insert_on_conflict_do_nothing(session, LsstDetection, detections_lsst_dict)


def insert_forced_sources(session, forced_sources: pd.DataFrame):
    if forced_sources.empty:
        return

    forced_detections_dict = forced_sources[FORCED_DETECTION_COLUMNS].to_dict("records")
    forced_detections_lsst_dict = forced_sources[
        [
            "oid",
            "sid",
            "measurement_id",
            "visit",
            "detector",
            "psfFlux",
            "psfFluxErr",
        ]
    ].to_dict("records")

    bulk_insert_on_conflict_do_nothing(session, ForcedPhotometry, forced_detections_dict)
    bulk_insert_on_conflict_do_nothing(session, LsstForcedPhotometry, forced_detections_lsst_dict)


def insert_non_detections(session, non_detections: pd.DataFrame):
    if non_detections.empty:
        return

    non_detections_lsst_dict = non_detections[
        [
            "oid",
            "sid",
            "ccdVisitId",
            "band",
            "mjd",
            "diaNoise",
        ]
    ].to_dict("records")

    bulk_insert_on_conflict_do_nothing(session, LsstNonDetection, non_detections_lsst_dict)