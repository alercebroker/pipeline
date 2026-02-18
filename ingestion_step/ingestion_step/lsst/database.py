import pandas as pd
from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.sql.models_pipeline import (
    Detection,
    ForcedPhotometry,
    LsstDetection,
    LsstDiaObject,
    LsstForcedPhotometry,
    LsstMpcOrbits,
    LsstSsDetection,
    Object,
)

from ingestion_step.core.database import (
    DETECTION_COLUMNS,
    FORCED_DETECTION_COLUMNS,
    OBJECT_COLUMNS,
    db_insert_on_conflict_do_nothing_builder,
    db_insert_on_conflict_do_update_builder,
)


def insert_dia_objects(
    driver: PsqlDatabase, dia_objects: pd.DataFrame, chunk_size: int | None = None
):
    if len(dia_objects) == 0:
        return
    if chunk_size is None:
        chunk_size = len(dia_objects)

    lsst_columns = (
        set(dia_objects)
        - set(OBJECT_COLUMNS)
        - {"mjd", "diaObjectId", "midpointMjdTai", "message_id"}
    )
    lsst_columns = lsst_columns.union({"oid"})

    objects_dict = dia_objects[OBJECT_COLUMNS].to_dict("records")
    objects_dia_lsst_dict = dia_objects[list(lsst_columns)].to_dict("records")

    with driver.session() as session:
        for i in range(0, len(dia_objects), chunk_size):
            objects_sql_stmt = db_insert_on_conflict_do_nothing_builder(
                Object,
                objects_dict[i : i + chunk_size],
            )
            objects_dia_lsst_sql_stmt = db_insert_on_conflict_do_update_builder(
                LsstDiaObject,
                objects_dia_lsst_dict[i : i + chunk_size],
                pk=["oid"],
            )

            session.execute(objects_sql_stmt)
            session.execute(objects_dia_lsst_sql_stmt)

        session.commit()


def insert_mpc_orbit(
    driver: PsqlDatabase, mpcorbs: pd.DataFrame, chunk_size: int | None = None
):
    if len(mpcorbs) == 0:
        return
    if chunk_size is None:
        chunk_size = len(mpcorbs)

    lsst_columns = set(mpcorbs.columns) - {"message_id", "midpointMjdTai", "mjd", "id"}

    mpcorbs_dict = mpcorbs[list(lsst_columns)].to_dict("records")

    with driver.session() as session:
        for i in range(0, len(mpcorbs), chunk_size):
            mpcorbs_sql_stmt = db_insert_on_conflict_do_update_builder(
                LsstMpcOrbits, mpcorbs_dict[i : i + chunk_size], pk=["ssObjectId"]
            )
            session.execute(mpcorbs_sql_stmt)
        session.commit()


def insert_fake_ss_objects(
    driver: PsqlDatabase, fake_ss_object: pd.DataFrame, chunk_size: int | None = None
):
    if len(fake_ss_object) == 0:
        return
    if chunk_size is None:
        chunk_size = len(fake_ss_object)

    lsst_columns = {
        "oid",
        "tid",
        "sid",
        "meanra",
        "meandec",
        "firstmjd",
        "lastmjd",
        "n_forced",
        "n_non_det",
    }

    fake_ss_object_dict = fake_ss_object[list(lsst_columns)].to_dict("records")

    with driver.session() as session:
        for i in range(0, len(fake_ss_object), chunk_size):
            fake_ss_object_sql_stmt = db_insert_on_conflict_do_nothing_builder(
                Object, fake_ss_object_dict[i : i + chunk_size]
            )
            session.execute(fake_ss_object_sql_stmt)
        session.commit()


def insert_sources(
    driver: PsqlDatabase,
    sources: pd.DataFrame,
    chunk_size: int | None = None,
    on_conflict_do_update: bool = False,
):
    if len(sources) == 0:
        return
    if chunk_size is None:
        chunk_size = len(sources)

    lsst_columns = (
        set(sources.columns)
        - set(DETECTION_COLUMNS)
        - {"message_id", "midpointMjdTai", "diaSourceId"}
    )
    lsst_columns = lsst_columns.union({"oid", "sid", "measurement_id"})

    detections_dict = sources[DETECTION_COLUMNS].to_dict("records")
    detections_lsst_dict = sources[list(lsst_columns)].to_dict("records")

    with driver.session() as session:
        for i in range(0, len(sources), chunk_size):
            if on_conflict_do_update:
                detections_sql_stmt = db_insert_on_conflict_do_update_builder(
                    Detection,
                    detections_dict[i : i + chunk_size],
                    pk=["oid", "measurement_id", "sid"],
                )

                detections_lsst_sql_stmt = db_insert_on_conflict_do_update_builder(
                    LsstDetection,
                    detections_lsst_dict[i : i + chunk_size],
                    pk=["oid", "measurement_id"],
                )
            else:
                detections_sql_stmt = db_insert_on_conflict_do_nothing_builder(
                    Detection,
                    detections_dict[i : i + chunk_size],
                )
                detections_lsst_sql_stmt = db_insert_on_conflict_do_nothing_builder(
                    LsstDetection,
                    detections_lsst_dict[i : i + chunk_size],
                )

            session.execute(detections_sql_stmt)
            session.execute(detections_lsst_sql_stmt)
        session.commit()


def insert_ss_sources(
    driver: PsqlDatabase, sources: pd.DataFrame, chunk_size: int | None = None
):
    if len(sources) == 0:
        return
    if chunk_size is None:
        chunk_size = len(sources)

    lsst_columns = set(sources.columns) - {
        "message_id",
        "midpointMjdTai",
        "parentDiaSourceId",
        "diaSourceId",
    }

    ss_sources_dict = sources[list(lsst_columns)].to_dict("records")

    with driver.session() as session:
        for i in range(0, len(sources), chunk_size):
            ss_source_sql_stmt = db_insert_on_conflict_do_nothing_builder(
                LsstSsDetection,
                ss_sources_dict[i : i + chunk_size],
            )

            session.execute(ss_source_sql_stmt)
        session.commit()


def insert_forced_sources(
    driver: PsqlDatabase, forced_sources: pd.DataFrame, chunk_size: int | None = None
):
    if len(forced_sources) == 0:
        return
    if chunk_size is None:
        chunk_size = len(forced_sources)

    lsst_columns = (
        set(forced_sources.columns)
        - set(FORCED_DETECTION_COLUMNS)
        - {"message_id", "diaForcedSourceId", "midpointMjdTai", "diaObjectId"}
    )
    lsst_columns = lsst_columns.union({"oid", "sid", "measurement_id"})

    forced_detections_dict = forced_sources[FORCED_DETECTION_COLUMNS].to_dict("records")
    forced_detections_lsst_dict = forced_sources[list(lsst_columns)].to_dict("records")

    with driver.session() as session:
        for i in range(0, len(forced_sources), chunk_size):
            forced_detections_sql_stmt = db_insert_on_conflict_do_nothing_builder(
                ForcedPhotometry,
                forced_detections_dict[i : i + chunk_size],
            )
            forced_detections_lsst_sql_stmt = db_insert_on_conflict_do_nothing_builder(
                LsstForcedPhotometry,
                forced_detections_lsst_dict[i : i + chunk_size],
            )

            session.execute(forced_detections_sql_stmt)
            session.execute(forced_detections_lsst_sql_stmt)
        session.commit()
