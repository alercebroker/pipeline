import copy
import logging
from abc import ABC, abstractmethod
from importlib.metadata import version
from io import StringIO
from typing import Dict, List

## En el import desde los models se deben traer todos los modelos que se modififiquen en los steps.
## Al menos desde correction hacia atras.
from db_plugins.db.sql.models import (
    Detection,
    Feature,
    ForcedPhotometry,
    LsstDiaObject,
    LsstMpcOrbits,
    MagStat,
    Object,
    Xmatch,
    ZtfDataquality,
    ZtfDetection,
    ZtfForcedPhotometry,
    ZtfGaia,
    ZtfObject,
    ZtfPS1,
    ZtfReference,
    ZtfSS,
    ProbabilityArchive,
)
from sqlalchemy import bindparam, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .parser import (
    parse_det,
    parse_fp,
    parse_obj_stats,
    parse_xmatch,
    parse_ztf_det,
    parse_ztf_dq,
    parse_ztf_fp,
    parse_ztf_gaia,
    parse_ztf_magstats,
    parse_ztf_object,
    parse_ztf_objstats,
    parse_ztf_ps1,
    parse_ztf_refernece,
    parse_ztf_ss,
    parse_probability,
)

step_version = version("scribe-multisurvey")


class Command(ABC):
    type: str

    def __init__(self, payload, criteria=None, options=None):
        self._check_inputs(payload, criteria)
        self.criteria = criteria or {}
        self.options = options
        self.data = self._format_data(payload)

    def _format_data(self, data):
        return data

    def _check_inputs(self, data, criteria):
        if not data:
            raise ValueError("Not data provided in command")

    @staticmethod
    @abstractmethod
    def db_operation(session: Session, data: List):
        pass


class ZTFCorrectionCommand(Command):
    type = "ZTFCorrectionCommand"

    def _format_data(self, data):
        """
        Generate a dictionary with  data for the tables:f
        ztf_detection
        ztf_forced photometry
        ps1_ztf
        ss_ztf
        gaia_ztf
        reference_ztf
        """

        oid = data["oid"]
        candidate_measurement_id = data["measurement_id"][0]

        candidate = {}
        ## TODO check potential issue, multiple candidates in the same message ? => Whats the solution?
        detections = []
        ztf_detections = []

        for detection in data["detections"]:
            if detection["new"]:
                if detection["measurement_id"] == candidate_measurement_id:
                    candidate = detection
                detections.append(parse_det(detection, oid))
                ztf_detections.append(parse_ztf_det(detection, oid))

        for detection in data["previous_detections"]:
            if detection["new"]:
                detections.append(parse_det(detection, oid))
                ztf_detections.append(parse_ztf_det(detection, oid))

        fp_dict = {}
        ztf_fp_dict = {}
        for forced in data["forced_photometries"]:
            if forced["new"]:
                fp = parse_fp(forced, oid)
                ztf_fp = parse_ztf_fp(forced, oid)

                key = (fp["oid"], fp["measurement_id"], fp["sid"])
                ztf_key = (ztf_fp["oid"], ztf_fp["measurement_id"])

                if key not in fp_dict or fp["mjd"] > fp_dict[key]["mjd"]:
                    fp_dict[key] = fp
                if (
                    ztf_key not in ztf_fp_dict
                    or ztf_fp["mjd"] > ztf_fp_dict[ztf_key]["mjd"]
                ):
                    ztf_fp_dict[ztf_key] = ztf_fp

        parsed_ps1 = parse_ztf_ps1(candidate, oid)
        parsed_ss = parse_ztf_ss(candidate, oid)
        parsed_gaia = parse_ztf_gaia(candidate, oid)
        parsed_dq = parse_ztf_dq(candidate, oid)
        parsed_reference = parse_ztf_refernece(candidate, oid)
        parsed_ztf_obj = parse_ztf_object(candidate, oid)

        return {
            "detections": detections,
            "ztf_detections": ztf_detections,
            "forced_photometries": list(fp_dict.values()),
            "ztf_forced_photometries": list(ztf_fp_dict.values()),
            "ps1": parsed_ps1,
            "ss": parsed_ss,
            "gaia": parsed_gaia,
            "dq": parsed_dq,
            "reference": parsed_reference,
            "ztf_object": parsed_ztf_obj,
        }

    @staticmethod
    def db_operation(session: Session, data: List):
        detections = []
        ztf_detections = []
        forced_photometries = []
        ztf_forced_photometries = []
        ps1 = []
        ss = []
        gaia = []
        dq = []
        reference = []
        ztf_obj_update = []

        for single_data in data:
            detections.extend(single_data["detections"])
            ztf_detections.extend(single_data["ztf_detections"])
            forced_photometries.extend(single_data["forced_photometries"])
            ztf_forced_photometries.extend(single_data["ztf_forced_photometries"])
            ps1.append(single_data["ps1"])
            ss.append(single_data["ss"])
            gaia.append(single_data["gaia"])
            dq.append(single_data["dq"])
            reference.append(single_data["reference"])
            ztf_obj_update.append(single_data["ztf_object"])

        fp_dedup_dict = {}
        for fp, ztf_fp in zip(forced_photometries, ztf_forced_photometries):
            key = (fp["oid"], fp["measurement_id"], fp["sid"])
            if key not in fp_dedup_dict or fp["mjd"] > fp_dedup_dict[key]["fp"]["mjd"]:
                fp_dedup_dict[key] = {"fp": fp, "ztf_fp": ztf_fp}

        forced_photometries = [item["fp"] for item in fp_dedup_dict.values()]
        ztf_forced_photometries = [item["ztf_fp"] for item in fp_dedup_dict.values()]

        det_dedup_dict = {}
        for det, ztf_det in zip(detections, ztf_detections):
            key = (det["oid"], det["measurement_id"], det["sid"])
            if (
                key not in det_dedup_dict
                or det["mjd"] > det_dedup_dict[key]["det"]["mjd"]
            ):
                det_dedup_dict[key] = {"det": det, "ztf_det": ztf_det}

        detections = [item["det"] for item in det_dedup_dict.values()]
        ztf_detections = [item["ztf_det"] for item in det_dedup_dict.values()]

        conn = session.connection()

        if detections:
            detections_stmt = insert(Detection).on_conflict_do_update(
                constraint="pk_detection_oid_measurementid_sid",
                set_=insert(Detection).excluded,
            )
            conn.execute(detections_stmt, detections)

        if ztf_detections:
            ztf_detections_stmt = insert(ZtfDetection).on_conflict_do_update(
                constraint="pk_ztfdetection_oid_measurementid",
                set_=insert(ZtfDetection).excluded,
            )
            conn.execute(ztf_detections_stmt, ztf_detections)

        if forced_photometries:
            fp_stmt = insert(ForcedPhotometry).on_conflict_do_update(
                constraint="pk_forcedphotometry_oid_measurementid_sid",
                set_=insert(ForcedPhotometry).excluded,
            )
            conn.execute(fp_stmt, forced_photometries)

        if ztf_forced_photometries:
            ztf_fp_stmt = insert(ZtfForcedPhotometry).on_conflict_do_update(
                constraint="pk_ztfforcedphotometry_oid_measurementid",
                set_=insert(ZtfForcedPhotometry).excluded,
            )
            conn.execute(ztf_fp_stmt, ztf_forced_photometries)

        ztfobj_dict = {}
        for obj in ztf_obj_update:
            key = obj["oid"]
            mjd = obj.get("_mjd", 0)
            if key not in ztfobj_dict or mjd > ztfobj_dict[key][0]:
                ztfobj_dict[key] = (mjd, obj)

        if ztfobj_dict:
            ztfobj_list = []
            for mjd, obj in ztfobj_dict.values():
                clean_obj = {k: v for k, v in obj.items() if k != "_mjd"}
                clean_obj["_oid"] = clean_obj["oid"]
                ztfobj_list.append(clean_obj)

            ztf_object_stmt = (
                update(ZtfObject)
                .where(ZtfObject.oid == bindparam("_oid"))
                .values(
                    {
                        "ndethist": bindparam("ndethist"),
                        "ncovhist": bindparam("ncovhist"),
                        "mjdstarthist": bindparam("mjdstarthist"),
                        "mjdendhist": bindparam("mjdendhist"),
                    }
                )
            )
            conn.execute(ztf_object_stmt, ztfobj_list)

        ps1_dict = {}
        for p in ps1:
            key = (p["oid"], p.get("measurement_id"))
            mjd = p.get("_mjd", 0)
            if key not in ps1_dict or mjd > ps1_dict[key][0]:
                ps1_dict[key] = (mjd, {k: v for k, v in p.items() if k != "_mjd"})

        if ps1_dict:
            ps1_clean = [obj for mjd, obj in ps1_dict.values()]
            ps_stmt = insert(ZtfPS1).on_conflict_do_update(
                constraint="pk_ztfps1_oid_measurement_id", set_=insert(ZtfPS1).excluded
            )
            conn.execute(ps_stmt, ps1_clean)

        ss_dict = {}
        for s in ss:
            key = (s["oid"], s.get("measurement_id"))
            mjd = s.get("_mjd", 0)
            if key not in ss_dict or mjd > ss_dict[key][0]:
                ss_dict[key] = (mjd, {k: v for k, v in s.items() if k != "_mjd"})

        if ss_dict:
            ss_clean = [obj for mjd, obj in ss_dict.values()]
            ss_stmt = insert(ZtfSS).on_conflict_do_update(
                constraint="pk_ztfss_oid_measurement_id", set_=insert(ZtfSS).excluded
            )
            conn.execute(ss_stmt, ss_clean)

        gaia_dict = {}
        for g in gaia:
            key = g["oid"]
            mjd = g.get("_mjd", 0)
            if key not in gaia_dict or mjd > gaia_dict[key][0]:
                gaia_dict[key] = (mjd, {k: v for k, v in g.items() if k != "_mjd"})

        if gaia_dict:
            gaia_clean = [obj for mjd, obj in gaia_dict.values()]
            gaia_stmt = insert(ZtfGaia).on_conflict_do_update(
                constraint="pk_ztfgaia_oid", set_=insert(ZtfGaia).excluded
            )
            conn.execute(gaia_stmt, gaia_clean)

        dq_dict = {}
        for d in dq:
            key = (d["oid"], d.get("measurement_id"))
            mjd = d.get("_mjd", 0)
            if key not in dq_dict or mjd > dq_dict[key][0]:
                dq_dict[key] = (mjd, {k: v for k, v in d.items() if k != "_mjd"})

        if dq_dict:
            dq_clean = [obj for mjd, obj in dq_dict.values()]
            dq_stmt = insert(ZtfDataquality).on_conflict_do_update(
                constraint="pk_ztfdataquality_oid_measurement_id",
                set_=insert(ZtfDataquality).excluded,
            )
            conn.execute(dq_stmt, dq_clean)

        ref_dict = {}
        for r in reference:
            key = (r["oid"], r.get("rfid"))
            mjd = r.get("_mjd", 0)
            if key not in ref_dict or mjd > ref_dict[key][0]:
                ref_dict[key] = (mjd, {k: v for k, v in r.items() if k != "_mjd"})

        if ref_dict:
            reference_clean = [obj for mjd, obj in ref_dict.values()]
            reference_stmt = insert(ZtfReference).on_conflict_do_update(
                constraint="pk_ztfreference_oid_rfid",
                set_=insert(ZtfReference).excluded,
            )
            conn.execute(reference_stmt, reference_clean)


class ZTFMagstatCommand(Command):
    type = "ZTFMagstatCommand"

    def _format_data(self, data):
        oid = data["oid"]
        sid = data["sid"]
        object_stats = parse_ztf_objstats(data, oid, sid)
        magstats_list = [
            parse_ztf_magstats(ms, oid, sid) for ms in data["magstats"]["0"]
        ]

        return {"object_stats": object_stats, "magstats": magstats_list}

    @staticmethod
    def db_operation(session: Session, data: List):
        objectstat_list = []
        dedup_magstats = {}  # We must deduplicate magstats and since none of its keys is good we will use  objstats stuff

        for single_data in data:
            obj = single_data["object_stats"]
            objectstat_list.append(obj)

            lastmjd = obj["lastmjd"]

            for ms in single_data["magstats"]:
                key = (ms["oid"], ms["sid"], ms["band"])

                if key not in dedup_magstats:
                    dedup_magstats[key] = (lastmjd, ms)
                else:
                    prev_lastmjd, _ = dedup_magstats[key]
                    if lastmjd > prev_lastmjd:
                        dedup_magstats[key] = (lastmjd, ms)

        magstat_list = [ms for _, ms in dedup_magstats.values()]

        # update object
        if len(objectstat_list) > 0:
            object_stmt = update(Object)
            object_result = session.connection().execute(
                object_stmt.where(Object.oid == bindparam("_oid")).values(
                    {
                        "meanra": bindparam("meanra"),
                        "meandec": bindparam("meandec"),
                        "sigmara": bindparam("sigmara"),
                        "sigmadec": bindparam("sigmadec"),
                        "firstmjd": bindparam("firstmjd"),
                        "lastmjd": bindparam("lastmjd"),
                        "deltamjd": bindparam("deltamjd"),
                        "n_det": bindparam("n_det"),
                        "n_forced": bindparam("n_forced"),
                        "n_non_det": bindparam("n_non_det"),
                    }
                ),
                objectstat_list,
            )

            # update ztf_object columns stellar and corrected
            ztf_object_stmt = update(ZtfObject)
            ztf_object_result = session.connection().execute(
                ztf_object_stmt.where(ZtfObject.oid == bindparam("_oid")).values(
                    {
                        "corrected": bindparam("corrected"),
                        "stellar": bindparam("stellar"),
                        "reference_change": bindparam("reference_change"),
                        "diffpos": bindparam("diffpos"),
                    }
                ),
                objectstat_list,
            )

        # insert magstats
        if len(magstat_list) > 0:
            magstats_stmt = insert(MagStat)
            magstats_result = session.connection().execute(
                magstats_stmt.on_conflict_do_update(
                    constraint="pk_magstat_oid_sid_band", set_=magstats_stmt.excluded
                ),
                magstat_list,
            )


class LSSTMagstatCommand(Command):
    type = "LSSTMagstatCommand"

    def _format_data(self, data):
        oid = data["oid"]
        sid = data["sid"]
        object_stats = parse_obj_stats(data, oid, sid)
        magstats_list = []

        return {"object_stats": object_stats, "magstats": magstats_list}

    @staticmethod
    def db_operation(session: Session, data: List):
        objectstat_list = []
        magstat_list = []

        for single_data in data:
            objectstat_list.append(single_data["object_stats"])
            magstat_list.extend(single_data["magstats"])

        # update object
        if len(objectstat_list) > 0:
            object_stmt = update(Object)
            object_result = session.connection().execute(
                object_stmt.where(Object.oid == bindparam("_oid")).values(
                    {
                        "meanra": bindparam("meanra"),
                        "meandec": bindparam("meandec"),
                        "sigmara": bindparam("sigmara"),
                        "sigmadec": bindparam("sigmadec"),
                        "firstmjd": bindparam("firstmjd"),
                        "lastmjd": bindparam("lastmjd"),
                        "deltamjd": bindparam("deltamjd"),
                        "n_det": bindparam("n_det"),
                        "n_forced": bindparam("n_forced"),
                        "n_non_det": bindparam("n_non_det"),
                    }
                ),
                objectstat_list,
            )


class LSSTFeatureCommand(Command):
    type = "LSSTFeatureCommand"

    def _format_data(self, data):
        oid = data["oid"]
        sid = data["sid"]
        feature_version = (
            data["features_version"].split(".")[0]
            if isinstance(data["features_version"], str)
            else data["features_version"]
        )

        features_tuples = []
        for feature in data["features"]:
            features_tuples.append(
                (
                    oid,
                    sid,
                    feature["feature_id"],
                    feature["band"],
                    feature["value"],
                    feature_version,
                )
            )
        return sorted(features_tuples)

    @staticmethod
    def db_operation(session: Session, data: List):
        if len(data) == 0:
            return

        buf = StringIO()
        for row in data:
            buf.write(",".join(map(str, row)) + "\n")
        buf.seek(0)

        raw_conn = session.connection().connection
        with raw_conn.cursor() as cur:
            # Create temp table (once per connection, persists)
            cur.execute("""
                CREATE TEMP TABLE IF NOT EXISTS staging_features (
                    oid BIGINT,
                    sid SMALLINT,
                    feature_id INT,
                    band SMALLINT,
                    value FLOAT8,
                    version INT
                ) ON COMMIT DROP
            """)

            cur.copy_from(
                buf,
                "staging_features",
                sep=",",
                columns=("oid", "sid", "feature_id", "band", "value", "version"),
            )

            # Batch UPDATE
            cur.execute("""
                UPDATE feature AS f
                SET value = s.value, version = s.version, updated_date = NOW()
                FROM staging_features AS s
                WHERE f.oid = s. oid AND f.sid = s.sid 
                AND f.feature_id = s.feature_id AND f.band = s.band
            """)

            # Batch INSERT
            cur.execute("""
                INSERT INTO feature (oid, sid, feature_id, band, value, version, updated_date)
                SELECT s. oid, s.sid, s.feature_id, s.band, s.value, s.version, NOW()
                FROM staging_features s
                LEFT JOIN feature f ON (
                    f.oid = s.oid AND f. sid = s.sid 
                    AND f.feature_id = s.feature_id AND f.band = s.band
                )
                WHERE f.oid IS NULL
            """)

            # Clear staging
            cur.execute("TRUNCATE staging_features")

            raw_conn.commit()


class XmatchCommand(Command):
    type = "XmatchCommand"

    def _format_data(self, data):
        parsed_xmatch = parse_xmatch(data)

        if parsed_xmatch is None:
            return None

        return {"xmatches": parsed_xmatch}
    
    @staticmethod
    def db_operation(session: Session, data: List):
        if len(data) > 0:
            dedup_dict = {}

            for item in data:
                if item is None:
                    continue
                else:
                    xmatch = item["xmatches"]
                    key = (xmatch["oid"], xmatch["sid"], xmatch["catalog_id"])
                    dedup_dict[key] = {
                        "oid": xmatch["oid"],
                        "sid": xmatch["sid"],
                        "catid": xmatch["catalog_id"],
                        "oid_catalog": xmatch["oid_catalog"],
                        "dist": xmatch["dist"],
                    }

            deduplicated_data = list(dedup_dict.values())

            if deduplicated_data:
                insert_stmt = insert(Xmatch).values(deduplicated_data)
                upsert_stmt = session.connection().execute(
                    insert_stmt.on_conflict_do_update(
                        index_elements=["oid", "sid", "catid"],
                        set_={
                            "oid_catalog": insert_stmt.excluded.oid_catalog,
                            "dist": insert_stmt.excluded.dist,
                        },
                    )
                )

class ProbabilityArchivalCommand(Command):
    type = "ProbabilityArchivalCommand"

    def _format_data(self, data):
        return parse_probability(data)

    @staticmethod
    def db_operation(session: Session, data: List):
        if not data:
            return

        dedup = {}
        for row in data:
            key = (row["oid"], row["sid"], row["classifier_id"], row["classifier_version"], row["class_id"])
            if key not in dedup or row["lastmjd"] > dedup[key]["lastmjd"]:
                dedup[key] = row

        records = list(dedup.values())

        stmt = insert(ProbabilityArchive)
        upsert = stmt.on_conflict_do_update(
            constraint="pk_probability_archive_oid_classifierid_version_classid",
            set_={
                "probability": stmt.excluded.probability,
                "ranking": stmt.excluded.ranking,
                "lastmjd": stmt.excluded.lastmjd,
            }
        )
        session.connection().execute(upsert, records)