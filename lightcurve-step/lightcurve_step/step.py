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
        self.last_mjd = {}

    @classmethod
    def pre_execute(cls, messages: List[dict]) -> dict:
        aids, detections, non_detections, oids = set(), [], [], {}
        last_mjds = {}
        for msg in messages:
            aid = msg["aid"]
            oids.update(
                {det["oid"]: aid for det in msg["detections"] if det["sid"] == "ztf"}
            )
            aids.add(aid)
            last_mjds[aid] = max(last_mjds.get(aid, 0), msg["detections"][0]["mjd"])
            detections.extend([det | {"new": True} for det in msg["detections"]])
            non_detections.extend(msg["non_detections"])
        logger = logging.getLogger("alerce.LightcurveStep")
        logger.debug(f"Received {len(detections)} detections from messages")
        return {
            "aids": list(aids),
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
        print(f'messages {len(messages["detections"])}')
        return {
            "detections": detections,
            "non_detections": non_detections,
            "last_mjds": messages["last_mjds"],
        }

    def _parse_ztf_detection(self, ztf_models: list, *, oids):
        fields = {
            "tid",
            "sid",
            "aid",
            "oid",
            "mjd",
            "fid",
            "ra",
            "sigmara",
            "dec",
            "sigmadec",
            "magpsf",
            "sigmapsf",
            "magpsf_corr",
            "sigmapsf_corr",
            "sigmapsf_corr_ext",
            "isdiffpos",
            "corrected",
            "dubious",
            "parent_candid",
            "has_stamp",
        }
        parsed_result = []
        for det in ztf_models:
            extra_fields = {}
            for field, value in det.items():
                if field not in fields and not field.startswith("_"):
                    extra_fields[field] = value
            parsed = MongoDetection(
                **det,
                aid=oids[det["oid"]],
                sid="ztf",
                tid="ztf",
                mag=det["magpsf"],
                e_mag=det["sigmapsf"],
                mag_corr=det["magpsf_corr"],
                e_mag_corr=det["sigmapsf_corr"],
                e_mag_corr_ext=det["sigmapsf_corr_ext"],
                e_ra=det["sigmara"],
                e_dec=det["sigmadec"],
                extra_fields=extra_fields,
            )
            parsed_result.append({**parsed, "forced": False, "new": False})

        return parsed_result

    def _parse_ztf_non_detection(self, ztf_models: list, *, oids):
        non_dets = []
        for non_det in ztf_models:
            non_dets.append(
                MongoNonDetection(
                    tid="ZTF",
                    sid="ZTF",
                    aid=oids[non_det["oid"]],
                    oid=non_det["oid"],
                    mjd=non_det["mjd"],
                    fid=non_det["fid"],
                    diffmaglims=non_det.get("diffmaglim", None),
                )
            )
        return non_dets

    def _parse_ztf_forced_photometry(self, ztf_models: list, *, oids):
        return [
            {
                **MongoForcedPhotometry(**forced, aid=oids[forced["oid"]]),
                "new": False,
                "forced": True,
            }
            for forced in ztf_models
        ]

    @classmethod
    def pre_produce(cls, result: dict) -> List[dict]:
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
            mjds = result["last_mjds"]
            dets = dets[dets["mjd"] <= mjds[aid]]
            output.append(
                {
                    "aid": aid,
                    "detections": dets.to_dict("records"),
                    "non_detections": nd,
                }
            )
        return output