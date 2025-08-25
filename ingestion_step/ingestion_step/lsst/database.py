import pandas as pd
from db_plugins.db.sql.models import (
    Detection,
    ForcedPhotometry,
    LsstDetection,
    LsstDiaObject,
    LsstForcedPhotometry,
    LsstMpcorb,
    LsstSsDetection,
    # LsstNonDetection,
    # LsstSsObject,
    Object,
)
from sqlalchemy.orm import Session

from ingestion_step.core.database import (
    DETECTION_COLUMNS,
    FORCED_DETECTION_COLUMNS,
    OBJECT_COLUMNS,
    db_statement_builder,
)


def insert_dia_objects(session: Session, dia_objects: pd.DataFrame):
    if len(dia_objects) == 0:
        return

    objects_dict = dia_objects[OBJECT_COLUMNS].to_dict("records")

    lsst_columns = (
        set(dia_objects)
        - set(OBJECT_COLUMNS)
        - {"mjd", "diaObjectId", "midpointMjdTai", "message_id"}
    )
    lsst_columns.union(
        {
            "oid",
            "sid",
        }
    )

    objects_dia_lsst_dict = dia_objects[list(lsst_columns)].to_dict("records")

    objects_sql_stmt = db_statement_builder(Object, objects_dict)
    objects_dia_lsst_sql_stmt = db_statement_builder(
        LsstDiaObject, objects_dia_lsst_dict
    )

    session.execute(objects_sql_stmt)
    session.execute(objects_dia_lsst_sql_stmt)


def insert_mpcorb(session: Session, mpcorbs: pd.DataFrame):
    if len(mpcorbs) == 0:
        return

    lsst_columns = set(mpcorbs.columns) - {
        "message_id",
    }

    mpcorbs_dict = mpcorbs[list(lsst_columns)].to_dict("records")

    mpcorbs_sql_stmt = db_statement_builder(LsstMpcorb, mpcorbs_dict)

    session.execute(mpcorbs_sql_stmt)


# def insert_ss_objects(session: Session, ss_objects: pd.DataFrame):
#     if len(ss_objects) == 0:
#         return
#     objects_dict = ss_objects[OBJECT_COLUMNS].to_dict("records")
#     objects_ss_lsst_dict = ss_objects[["oid"]].to_dict("records")
#
#     objects_sql_stmt = db_statement_builder(Object, objects_dict)
#     objects_ss_lsst_sql_stmt = db_statement_builder(LsstSsObject, objects_ss_lsst_dict)
#
#     with driver.session() as session:
#         session.execute(objects_sql_stmt)
#         session.execute(objects_ss_lsst_sql_stmt)
#         session.commit()


def insert_sources(session: Session, sources: pd.DataFrame):
    if len(sources) == 0:
        return
    detections_dict = sources[DETECTION_COLUMNS].to_dict("records")

    lsst_columns = (
        set(sources.columns) - set(DETECTION_COLUMNS) - {"message_id", "midpointMjdTai"}
    )
    lsst_columns = lsst_columns.union({"oid", "sid", "measurement_id"})

    detections_lsst_dict = sources[list(lsst_columns)].to_dict("records")

    detections_sql_stmt = db_statement_builder(Detection, detections_dict)
    detections_lsst_sql_stmt = db_statement_builder(LsstDetection, detections_lsst_dict)

    session.execute(detections_sql_stmt)
    session.execute(detections_lsst_sql_stmt)


def insert_ss_sources(session: Session, sources: pd.DataFrame):
    if len(sources) == 0:
        return

    lsst_columns = set(sources.columns) - {
        "message_id",
    }

    ss_sources_dict = sources[list(lsst_columns)].to_dict("records")

    ss_source_sql_stmt = db_statement_builder(LsstSsDetection, ss_sources_dict)

    session.execute(ss_source_sql_stmt)


def insert_forced_sources(session: Session, forced_sources: pd.DataFrame):
    if len(forced_sources) == 0:
        return
    forced_detections_dict = forced_sources[FORCED_DETECTION_COLUMNS].to_dict("records")

    lsst_columns = (
        set(forced_sources.columns)
        - set(FORCED_DETECTION_COLUMNS)
        - {"message_id", "diaForcedSourceId", "midpointMjdTai", "diaObjectId"}
    )
    lsst_columns = lsst_columns.union({"oid", "sid", "measurement_id"})

    forced_detections_lsst_dict = forced_sources[list(lsst_columns)].to_dict("records")

    forced_detections_sql_stmt = db_statement_builder(
        ForcedPhotometry, forced_detections_dict
    )
    forced_detections_lsst_sql_stmt = db_statement_builder(
        LsstForcedPhotometry, forced_detections_lsst_dict
    )

    session.execute(forced_detections_sql_stmt)
    session.execute(forced_detections_lsst_sql_stmt)


# def insert_non_detections(session: Session, non_detections: pd.DataFrame):
#     if len(non_detections) == 0:
#         return
#     non_detections_lsst_dict = non_detections[
#         [
#             "oid",
#             "sid",
#             "ccdVisitId",
#             "band",
#             "mjd",
#             "diaNoise",
#         ]
#     ].to_dict("records")
#
#     non_detections_sql_stmt = db_statement_builder(
#         LsstNonDetection, non_detections_lsst_dict
#     )
#
#     with driver.session() as session:
#         session.execute(non_detections_sql_stmt)
#         session.commit()
