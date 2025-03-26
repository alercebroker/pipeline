from typing import Any, List

from db_plugins.db.sql.models import (
    Detection,
    ForcedPhotometry,
    NonDetection,
    Object,
    ZtfDetection,
    ZtfForcedPhotometry,
    ZtfObject,
)
from sqlalchemy.dialects.postgresql import insert

from ..database import PsqlConnection


def insert_empty_objects_to_sql(db: PsqlConnection, records: List[dict[str, Any]]):
    # insert into db values = records on conflict do nothing
    def format_extra_fields(record: dict[str, Any]):
        extra_fields = record["extra_fields"]
        return {
            "ndethist": extra_fields["ndethist"],
            "ncovhist": extra_fields["ncovhist"],
            "mjdstarthist": extra_fields["jdstarthist"] - 2400000.5,
            "mjdendhist": extra_fields["jdendhist"] - 2400000.5,
            "meanra": record["ra"],
            "meandec": record["dec"],
            "firstmjd": record["mjd"],
            "lastmjd": record["mjd"],
            "deltajd": 0,
            "step_id_corr": "",
        }

    oids = {
        r["_id"]: format_extra_fields(r) for r in records if r["sid"].lower() == "ztf"
    }
    with db.session() as session:
        to_insert = [{"oid": oid, **extra_fields} for oid, extra_fields in oids.items()]
        statement = insert(Object).values(to_insert)
        statement = statement.on_conflict_do_update(
            "object_pkey",
            set_=dict(
                ndethist=statement.excluded.ndethist,
                ncovhist=statement.excluded.ncovhist,
                mjdstarthist=statement.excluded.mjdstarthist,
                mjdendhist=statement.excluded.mjdendhist,
            ),
        )
        session.execute(statement)
        session.commit()


def _db_statement_builder(model, data):
    stmt = insert(model).values(data).on_conflict_do_nothing()
    return stmt


def insert_objects(connection, objects_df):
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

    objects_df_parsed = objects_df[
        [
            "oid",
            "tid",
            "sid",
            "meanra",
            "meandec",
            "sigmara",
            "sigmadec",
            "firstmjd",
            "lastmjd",
            "deltamjd",
            "n_det",
            "n_forced",
            "n_non_det",
            "corrected",
            "stellar",
        ]
    ]
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

    object_sql_stmt = _db_statement_builder(Object, objects_dict)
    object_ztf_sql_stmt = _db_statement_builder(ZtfObject, objects_ztf_dict)

    with connection.session() as session:
        session.execute(object_sql_stmt)
        session.execute(object_ztf_sql_stmt)
        session.commit()


def insert_detections(connection, detections_df):
    # editing the df
    detections_df["magpsf_corr"] = None
    detections_df["sigmapsf_corr"] = None
    detections_df["sigmapsf_corr_ext"] = None
    detections_df["corrected"] = False
    detections_df["dubious"] = False
    detections_df["has_stamp"] = True

    detections_df = detections_df.reset_index()

    detections_df_parsed = detections_df[
        [
            "oid",
            "measurement_id",
            "mjd",
            "ra",
            "dec",
            "band",
        ]
    ]
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
            "magapfloat4",
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

    detection_sql_stmt = _db_statement_builder(Detection, detections_dict)
    detection_ztf_sql_stmt = _db_statement_builder(ZtfDetection, detections_ztf_dict)

    with connection.session() as session:
        session.execute(detection_sql_stmt)
        session.execute(detection_ztf_sql_stmt)
        session.commit()


def insert_forced_photometry(connection, forced_photometry_df):
    # editing the df
    forced_photometry_df["mag_corr"] = None
    forced_photometry_df["e_mag_corr"] = None
    forced_photometry_df["e_mag_corr_ext"] = None
    forced_photometry_df["isdiffpos"] = -1
    forced_photometry_df["corrected"] = False
    forced_photometry_df["dubious"] = False
    forced_photometry_df["has_stamp"] = True

    forced_photometry_df = forced_photometry_df.reset_index()

    forced_photometry_df_parsed = forced_photometry_df[
        [
            "oid",
            "measurement_id",
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
    forced_photometry_dict = forced_photometry_df_parsed.to_dict("records")
    forced_photometry_ztf_df_parsed = forced_photometry_df[
        [
            "oid",
            "measurement_id",
            "mjd",
            "ra",
            "dec",
            "band",
        ]
    ]
    forced_photometry_ztf_dict = forced_photometry_ztf_df_parsed.to_dict("records")

    forced_photometry_sql_stmt = _db_statement_builder(
        ForcedPhotometry, forced_photometry_dict
    )
    forced_photometry_ztf_sql_stmt = _db_statement_builder(
        ZtfForcedPhotometry, forced_photometry_ztf_dict
    )

    with connection.session() as session:
        session.execute(forced_photometry_sql_stmt)
        session.execute(forced_photometry_ztf_sql_stmt)
        session.commit()


def insert_non_detections(connection, non_detections_df):
    non_detections_df = non_detections_df.reset_index()
    non_detections_dict = non_detections_df[
        [
            "oid",
            "band",
            "mjd",
            "diffmaglim",
        ]
    ].to_dict("records")

    non_detection_sql_stmt = _db_statement_builder(NonDetection, non_detections_dict)

    with connection.session() as session:
        session.execute(non_detection_sql_stmt)
        session.commit()
