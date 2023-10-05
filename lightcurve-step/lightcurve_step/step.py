import logging
import numpy as np
import pandas as pd
import pickle
from typing import List
from apf.core.step import GenericStep
from .database_mongo import DatabaseConnection
from .database_sql import PSQLConnection
from sqlalchemy import select, text
from db_plugins.db.sql.models import Detection, NonDetection
from db_plugins.db.mongo.models import Detection as MongoDetection, NonDetection as MongoNonDetection


DETECTION = "detection"
NON_DETECTION = "non_detection"
FORCED_PHOTOMETRY = "phorced_photometry"


class LightcurveStep(GenericStep):
    def __init__(self, config: dict, db_mongo: DatabaseConnection, db_sql: PSQLConnection , **kwargs):
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
            oids.update((det["oid"],aid) for det in msg["detections"] if det["sid"] == "ztf")
            aids.add(aid)
            last_mjds[aid] = max(last_mjds.get(aid, 0), msg["detections"][0]["mjd"])
            detections.extend([det | {"new": True} for det in msg["detections"]])
            non_detections.extend(msg["non_detections"])
        logger = logging.getLogger("alerce.LightcurveStep")
        logger.debug(f"Received {len(detections)} detections from messages")
        return {
            "aids": aids,
            "oids": oids,
            "last_mjds": last_mjds,
            "detections": detections,
            "non_detections": non_detections,
        }

    def execute(self, messages: dict) -> dict:
        """Queries the database for all detections and non-detections for each AID and removes duplicates"""


        aids = list(messages["aids"])
        oids = list(messages["oids"])
        db_mongo_detections = self._get_mongo_detections(aids)
        db_mongo_non_detections = self._get_mongo_non_detections(aids)
        db_mongo_forced_photometries = self._get_mongo_forced_photometries(aids)
        db_sql_detections = self._get_sql_detections(oids)
        db_sql_non_detections = self._get_sql_non_detections(oids)
        #db_sql_forced_photometries = self._get_sql_forced_photometries(oids)        
        detections = pd.DataFrame(
            messages["detections"] + list(db_mongo_detections) + list(db_sql_detections) + list(db_mongo_forced_photometries)
        )
        non_detections = pd.DataFrame(
            messages["non_detections"] + list(db_mongo_non_detections) + list(db_sql_non_detections)
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
            "detections": detections,
            "non_detections": non_detections,
            "last_mjds": messages["last_mjds"]
        }

    def _ztf_to_mst(ztf_model, format, aid):

        if format == "Detection":
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
            extra_fields = {}
            for field, value in ztf_model.items():
                if field not in fields and not field.startswith("_"):
                    extra_fields[field] = value
            return MongoDetection(
                **ztf_model,
                sid = "ztf",
                tid = "ztf",
                aid = aid,
                mag = ztf_model["magpsf"],
                e_mag = ztf_model["sigmapsf"],
                mag_corr = ztf_model["magpsf_corr"],
                e_mag_corr = ztf_model["sigmapsf_corr"],
                e_mag_corr_ext = ztf_model["sigmapsf_corr_ext"],
                e_ra = ztf_model["sigmara"],
                e_dec = ztf_model["sigmadec"],
                extra_fields = extra_fields,

            )
        elif format == "NonDetection":
            return MongoNonDetection(
                tid = "ztf",
                sid = "ztf",
                aid = aid,
                oid = ztf_model["oid"],
                mjd = ztf_model["mjd"],
                fid = ztf_model["fid"],
                diffmaglims=ztf_model.get("diffmaglim", None),
            )
        elif format == "Forced":
            pass

        return 

    def _get_sql_detections(self,oids):
        results = []
        for oid in oids:
            with self.db_sql.session() as session:
                stmt = select(Detection, text("'ztf'")).filter(
                    Detection.oid == oid
                )
                results.append(self._ztf_to_mst(session.execute(stmt), format = "Detection", aid = oids[oid])) 

        return results
    
    def _get_sql_non_detections(self,oids):
        results = []
        for oid in oids:
            with self.db_sql.session() as session:
                stmt = select(NonDetection, text("'ztf'")).filter(
                    NonDetection.oid == oid
                )
                results.append(self._ztf_to_mst(session.execute(stmt), format = "NonDetection", aid = oids[oid])) 

        return results

    def _get_sql_forced_photometries(self,oids):
        return
    
    def _get_mongo_detections(self,aids):
        db_detections = self.db_mongo.database[DETECTION].aggregate(
            [
                {"$match": {"aid": {"$in": aids}}},
                {
                    "$addFields": {
                        "candid": "$_id",
                        "forced": False,
                        "new": False,
                    }
                },
                {"$project": {"_id": False, "evilDocDbHack": False}},
            ]
        )
        return db_detections
    
    def _get_mongo_non_detections(self,aids):
        db_non_detections = self.db_mongo.database[NON_DETECTION].find(
            {"aid": {"$in": aids}},
            {"_id": False, "evilDocDbHack": False},
        )
        return db_non_detections

    def _get_mongo_forced_photometries(self,aids):
        db_forced_photometries = self.db_mongo.database[FORCED_PHOTOMETRY].aggregate(
            [
                {"$match": {"aid": {"$in": aids}}},
                {
                    "$addFields": {
                        "candid": "$_id",
                        "forced": True,
                        "new": False,
                    }
                },
                {"$project": {"_id": False, "evilDocDbHack": False}},
            ]
        )
        return db_forced_photometries


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
