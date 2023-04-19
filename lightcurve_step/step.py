import logging
from typing import List

import numpy as np
import pandas as pd
from apf.core.step import GenericStep
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.mongo.models import Detection, NonDetection

# TODO: Unnecessary when using a clean DB
FID_MAPPING = {
    1: "g",
    2: "r",
    3: "i",
    4: "c",
    5: "o",
    6: "H",
}


class LightcurveStep(GenericStep):
    def __init__(
        self, config: dict, db_client: MongoConnection, level: int = logging.INFO
    ):
        super().__init__(config=config, level=level)
        self.db_client = db_client
        self.db_client.connect(config["DB_CONFIG"])

    @classmethod
    def pre_execute(cls, messages: List[dict]) -> dict:
        aids, detections, non_detections = set(), [], []
        for msg in messages:
            aids.add(msg["aid"])
            detections.extend(msg["detections"])
            non_detections.extend(msg["non_detections"])
        return {
            "aids": aids,
            "detections": detections,
            "non_detections": non_detections,
        }

    def execute(self, messages: dict) -> dict:
        """Queries the database for all detections and non-detections for each AID and removes duplicates"""
        query_detections = self.db_client.query(Detection)
        query_non_detections = self.db_client.query(NonDetection)

        # TODO: when using clean DB addFields step should be: {$addFields: {"candid": "$_id"}}
        db_detections = query_detections.collection.aggregate(
            [
                {"$match": {"aid": {"$in": list(messages["aids"])}}},
                {
                    "$addFields": {
                        "candid": {
                            "$cond": {
                                "if": "$candid",
                                "then": "$candid",
                                "else": "$_id",
                            }
                        },
                        "pid": {
                            "$cond": {
                                "if": "$pid",
                                "then": "$pid",
                                "else": 1,
                            }
                        },
                        "has_stamp": {
                            "$cond": {
                                "if": "$has_stamp",
                                "then": "$has_stamp",
                                "else": True,
                            }
                        },
                    }
                },
                {"$project": {"_id": False}},
            ]
        )
        db_non_detections = query_non_detections.collection.find(
            {"aid": {"$in": list(messages["aids"])}}, {"_id": False}
        )

        detections = pd.DataFrame(messages["detections"] + list(db_detections))
        non_detections = pd.DataFrame(
            messages["non_detections"] + list(db_non_detections)
        )

        detections = detections.sort_values(
            "has_stamp", ascending=False
        ).drop_duplicates("candid", keep="first")
        non_detections = non_detections.drop_duplicates(["oid", "fid", "mjd"])

        if detections["sid"].isna().any():  # TODO: Remove when using clean DB
            detections["sid"][detections["tid"] == "ZTF"] = "ZTF"
            detections["sid"][detections["tid"].str.startswith("ATLAS")] = "ATLAS"
            non_detections["sid"][non_detections["tid"] == "ZTF"] = "ZTF"
            non_detections["sid"][non_detections["tid"].str.startswith("ATLAS")] = "ATLAS"

        detections["fid"].replace(FID_MAPPING, inplace=True)  # TODO: Remove when using clean DB
        non_detections["fid"].replace(FID_MAPPING, inplace=True)  # TODO: Remove when using clean DB

        return {
            "detections": detections.replace(np.nan, None).to_dict("records"),
            "non_detections": non_detections.replace(np.nan, None).to_dict("records"),
        }

    @classmethod
    def pre_produce(cls, result: dict) -> List[dict]:
        detections = pd.DataFrame(result["detections"]).groupby("aid")
        try:  # At least one non-detection
            non_detections = pd.DataFrame(result["non_detections"]).groupby("aid")
        except (
            KeyError
        ):  # to reproduce expected error for missing non-detections in loop
            non_detections = pd.DataFrame(columns=["aid"]).groupby("aid")
        output = []
        for aid, dets in detections:
            try:
                nd = non_detections.get_group(aid).to_dict("records")
            except KeyError:
                nd = []
            output.append(
                {
                    "aid": aid,
                    "detections": dets.to_dict("records"),
                    "non_detections": nd,
                }
            )
        return output
