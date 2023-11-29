import logging
import numpy as np
import pandas as pd
import pickle
from typing import List
from apf.core.step import GenericStep
from .database_mongo import (
    DatabaseConnection,
    _get_mongo_non_detections,
    _get_mongo_detections,
    _get_mongo_forced_photometries,
)
from .database_sql import (
    PSQLConnection,
    _get_sql_detections,
    _get_sql_forced_photometries,
    _get_sql_non_detections,
)
from db_plugins.db.mongo.models import (
    Detection as MongoDetection,
    NonDetection as MongoNonDetection,
    ForcedPhotometry as MongoForcedPhotometry,
)


class LightcurveStep(GenericStep):
    def __init__(
        self,
        config: dict,
        db_mongo: DatabaseConnection,
        db_sql: PSQLConnection,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.db_mongo = db_mongo
        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.LightcurveStep")
        self.set_producer_key_field("aid")

    def pre_execute(self, messages: List[dict]) -> dict:
        aids, detections, non_detections, oids = set(), [], [], {}
        last_mjds = {}
        candids = {}
        for msg in messages:
            aid = msg["aid"]
            if aid not in candids:
                candids[aid] = []
            candids[aid].append(str(msg["candid"]))
            oids.update(
                {
                    det["oid"]: aid
                    for det in msg["detections"]
                    if det["sid"].lower() == "ztf"
                }
            )
            aids.add(aid)
            last_mjds[aid] = max(last_mjds.get(aid, 0), msg["detections"][0]["mjd"])
            detections.extend([det | {"new": True} for det in msg["detections"]])
            non_detections.extend(msg["non_detections"])

        logger = logging.getLogger("alerce.LightcurveStep")
        logger.debug(f"Received {len(detections)} detections from messages")
        return {
            "aids": list(aids),
            "candids": candids,
            "oids": oids,
            "last_mjds": last_mjds,
            "detections": detections,
            "non_detections": non_detections,
        }

    def execute(self, messages: dict) -> dict:
        """Queries the database for all detections and non-detections for each AID and removes duplicates"""

        db_mongo_detections = _get_mongo_detections(messages["aids"], self.db_mongo)
        db_mongo_non_detections = _get_mongo_non_detections(
            messages["aids"], self.db_mongo
        )
        db_mongo_forced_photometries = _get_mongo_forced_photometries(
            messages["aids"], self.db_mongo
        )
        db_sql_detections = _get_sql_detections(
            messages["oids"], self.db_sql, self._parse_ztf_detection
        )
        db_sql_non_detections = _get_sql_non_detections(
            messages["oids"], self.db_sql, self._parse_ztf_non_detection
        )
        db_sql_forced_photometries = _get_sql_forced_photometries(
            messages["oids"], self.db_sql, self._parse_ztf_forced_photometry
        )
        detections = pd.DataFrame(
            messages["detections"]
            + list(db_mongo_detections)
            + list(db_sql_detections)
            + list(db_mongo_forced_photometries)
            + list(db_sql_forced_photometries)
        )
        non_detections = pd.DataFrame(
            messages["non_detections"]
            + list(db_mongo_non_detections)
            + list(db_sql_non_detections)
        )
        self.logger.debug(f"Retrieved {detections.shape[0]} detections")
        detections["candid"] = detections["candid"].astype(str)
        detections["parent_candid"] = detections["parent_candid"].astype(str)

        # Try to keep those with stamp coming from the DB if there are clashes
        # maybe drop duplicates with candid and AID in LSST/ELAsTiCC
        detections = detections.sort_values(
            ["has_stamp", "new"], ascending=[False, True]
        ).drop_duplicates("candid", keep="first")

        non_detections = non_detections.drop_duplicates(["aid", "fid", "mjd"])
        self.logger.debug(
            f"Obtained {len(detections[detections['new']])} new detections"
        )
        return {
            "candids": messages["candids"],
            "detections": detections,
            "non_detections": non_detections,
            "last_mjds": messages["last_mjds"],
        }

    def _parse_ztf_detection(self, ztf_models: list, *, oids):
        GENERIC_FIELDS = {
            "tid",
            "sid",
            "aid",
            "oid",
            "pid",
            "mjd",
            "fid",
            "ra",
            "dec",
            "isdiffpos",
            "corrected",
            "dubious",
            "candid",
            "parent_candid",
            "has_stamp",
        }
        FID = {1: "g", 2: "r", 0: None, 12: "gr"}

        parsed_result = []
        for det in ztf_models:
            det: dict = det[0].__dict__
            extra_fields = {}
            parsed_det = {}
            for field, value in det.items():
                if field.startswith("_"):
                    continue
                if field not in GENERIC_FIELDS:
                    extra_fields[field] = value
                if field == "fid":
                    parsed_det[field] = FID[value]
                else:
                    parsed_det[field] = value
            parsed = MongoDetection(
                **parsed_det,
                aid=oids[det["oid"]],
                sid="ZTF",
                tid="ZTF",
                mag=det["magpsf"],
                e_mag=det["sigmapsf"],
                mag_corr=det["magpsf_corr"],
                e_mag_corr=det["sigmapsf_corr"],
                e_mag_corr_ext=det["sigmapsf_corr_ext"],
                extra_fields=extra_fields,
                e_ra=-999,
                e_dec=-999,
            )
            parsed["candid"] = parsed.pop("_id")
            parsed_result.append({**parsed, "forced": False, "new": False})

        return parsed_result

    def _parse_ztf_non_detection(self, ztf_models: list, *, oids):
        FID = {1: "g", 2: "r", 0: None, 12: "gr"}
        non_dets = []
        for non_det in ztf_models:
            non_det = non_det[0].__dict__
            mongo_non_detection = MongoNonDetection(
                _id="jej",
                tid="ZTF",
                sid="ZTF",
                aid=oids[non_det["oid"]],
                oid=non_det["oid"],
                mjd=non_det["mjd"],
                fid=FID[non_det["fid"]],
                diffmaglim=non_det.get("diffmaglim", None),
            )
            mongo_non_detection.pop("_id")
            non_dets.append(mongo_non_detection)
        return non_dets

    def _parse_ztf_forced_photometry(self, ztf_models: list, *, oids):
        def format_as_detection(fp):
            FID = {1: "g", 2: "r", 0: None, 12: "gr"}
            fp["fid"] = FID[fp["fid"]]
            fp["e_ra"] = 0
            fp["e_dec"] = 0
            fp["candid"] = fp.pop("_id")
            fp["extra_fields"] = {
                k: v for k, v in fp["extra_fields"].items() if not k.startswith("_")
            }
            # remove problematic fields
            FIELDS_TO_REMOVE = [
                "stellar",
                "e_mag_corr",
                "corrected",
                "mag_corr",
                "e_mag_corr_ext",
                "dubious",
            ]
            for field in FIELDS_TO_REMOVE:
                fp.pop(field, None)
            return fp

        parsed = [
            {
                **MongoForcedPhotometry(
                    **forced[0].__dict__,
                    aid=oids[forced[0].__dict__["oid"]],
                    sid="ZTF",
                    tid="ZTF",
                    candid=forced[0].__dict__["oid"] + str(forced[0].__dict__["pid"]),
                ),
                "new": False,
                "forced": True,
            }
            for forced in ztf_models
        ]

        return list(map(format_as_detection, parsed))

    def pre_produce(self, result: dict) -> List[dict]:
        def serialize_dia_object(ef: dict):
            if "diaObject" not in ef or not isinstance(ef["diaObject"], list):
                return ef

            ef["diaObject"] = pickle.dumps(ef["diaObject"])
            return ef

        detections = result["detections"].replace(np.nan, None).groupby("aid")
        try:  # At least one non-detection
            non_detections = (
                result["non_detections"].replace(np.nan, None).groupby("aid")
            )
        except KeyError:  # Handle empty non-detections
            non_detections = pd.DataFrame(columns=["aid"]).groupby("aid")
        output = []
        for aid, dets in detections:
            try:
                nd = non_detections.get_group(aid).to_dict("records")
            except KeyError:
                nd = []
            dets["extra_fields"] = dets["extra_fields"].apply(serialize_dia_object)
            if not self.config["FEATURE_FLAGS"].get("SKIP_MJD_FILTER", False):
                mjds = result["last_mjds"]
                dets = dets[dets["mjd"] <= mjds[aid]]
            output.append(
                {
                    "aid": aid,
                    "candid": result["candids"][aid],
                    "detections": dets.to_dict("records"),
                    "non_detections": nd,
                }
            )
        self.candid_found = False
        return output
