import logging
import numpy as np
import pandas as pd
import pickle
from typing import List
from apf.core.step import GenericStep
from .database import DatabaseConnection

DETECTION = "detection"
NON_DETECTION = "non_detection"
FORCED_PHOTOMETRY = "phorced_photometry"


class LightcurveStep(GenericStep):
    def __init__(self, config: dict, db_client: DatabaseConnection, **kwargs):
        super().__init__(config=config, **kwargs)
        self.db_client = db_client
        self.logger = logging.getLogger("alerce.LightcurveStep")
        self.last_mjd = {}

    @classmethod
    def pre_execute(cls, messages: List[dict]) -> dict:
        aids, detections, non_detections = set(), [], []
        last_mjds = {}
        for msg in messages:
            aid = msg["aid"]
            aids.add(aid)
            last_mjds[aid] = max(last_mjds.get(aid, 0), msg["detections"][0]["mjd"])
            detections.extend([det | {"new": True} for det in msg["detections"]])
            non_detections.extend(msg["non_detections"])

        logger = logging.getLogger("alerce.LightcurveStep")
        logger.debug(f"Received {len(detections)} detections from messages")
        return {
            "aids": aids,
            "last_mjds": last_mjds,
            "detections": detections,
            "non_detections": non_detections,
        }

    def execute(self, messages: dict) -> dict:
        """Queries the database for all detections and non-detections for each AID and removes duplicates"""
        db_detections = self.db_client.database[DETECTION].aggregate(
            [
                {"$match": {"aid": {"$in": list(messages["aids"])}}},
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
        db_non_detections = self.db_client.database[NON_DETECTION].find(
            {"aid": {"$in": list(messages["aids"])}},
            {"_id": False, "evilDocDbHack": False},
        )

        db_forced_photometries = self.db_client.database[FORCED_PHOTOMETRY].aggregate(
            [
                {"$match": {"aid": {"$in": list(messages["aids"])}}},
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

        detections = pd.DataFrame(
            messages["detections"] + list(db_detections) + list(db_forced_photometries)
        )
        non_detections = pd.DataFrame(
            messages["non_detections"] + list(db_non_detections)
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
            "last_mjds": messages["last_mjds"],
        }

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
