import logging
import numpy as np
import pandas as pd
import pickle
from typing import List
from apf.core.step import GenericStep
from db_plugins.db.mongo.connection import MongoConnection
from db_plugins.db.mongo.models import Detection, NonDetection, ForcedPhotometry


class LightcurveStep(GenericStep):
    def __init__(self, config: dict, db_client: MongoConnection, **kwargs):
        super().__init__(config=config, **kwargs)
        self.db_client = db_client
        self.db_client.connect(config["DB_CONFIG"])
        self.logger = logging.getLogger("alerce.LightcurveStep")

    @classmethod
    def pre_execute(cls, messages: List[dict]) -> dict:
        aids, detections, non_detections = set(), [], []
        for msg in messages:
            aids.add(msg["aid"])
            detections.extend([det | {"new": True} for det in msg["detections"]])
            non_detections.extend(msg["non_detections"])
        logger = logging.getLogger("alerce.LightcurveStep")
        logger.debug(f"Received {len(detections)} detections from messages")
        return {
            "aids": aids,
            "detections": detections,
            "non_detections": non_detections,
        }

    def execute(self, messages: dict) -> dict:
        """Queries the database for all detections and non-detections for each AID and removes duplicates"""
        query_detections = self.db_client.query(Detection)
        query_non_detections = self.db_client.query(NonDetection)
        query_forced_photometries = self.db_client.query(ForcedPhotometry)

        db_detections = query_detections.collection.aggregate(
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
        db_non_detections = query_non_detections.collection.find(
            {"aid": {"$in": list(messages["aids"])}},
            {"_id": False, "evilDocDbHack": False},
        )

        db_forced_photometries = query_forced_photometries.collection.aggregate(
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
            output.append(
                {
                    "aid": aid,
                    "detections": dets.to_dict("records"),
                    "non_detections": nd,
                }
            )
        return output
