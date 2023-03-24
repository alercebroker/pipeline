import logging
from typing import List

from apf.core.step import GenericStep
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.mongo.models import Detection, NonDetection


class LightcurveStep(GenericStep):
    def __init__(
        self, config: dict, db_client: MongoConnection, level: int = logging.INFO
    ):
        super().__init__(config=config, level=level)
        self.db_client = db_client

    @staticmethod
    def unique_detections(old_detections, new_detections):
        """Return only non-duplicate detections (based on candid). Keeps the ones in old"""
        candids = [det["candid"] for det in old_detections]

        new_detections = [det for det in new_detections if det["candid"] not in candids]
        return old_detections + new_detections

    @staticmethod
    def unique_non_detections(old_non_detections, new_non_detections):
        """Return only non-duplicate non-detections (based on oid, fid and mjd). Keeps the ones in old"""

        def create_id(detection):
            return {k: v for k, v in detection.items() if k in ["oid", "fid", "mjd"]}

        ids = [create_id(det) for det in old_non_detections]

        new_non_detections = [
            det for det in new_non_detections if create_id(det) not in ids
        ]
        return old_non_detections + new_non_detections

    def pre_execute(self, messages: List[dict]):
        """If multiple AIDs in the same batch create a single message with all of them"""
        aids, output = {}, []
        for message in messages:
            if message["aid"] in aids:
                idx = aids[message["aid"]]
                output[idx]["detections"] = self.unique_detections(
                    output[idx]["detections"], message["detections"]
                )
                output[idx]["non_detections"] = self.unique_non_detections(
                    output[idx]["non_detections"], message["non_detections"]
                )
            else:
                output.append(message)
                aids[message["aid"]] = len(output) - 1
        return output

    def execute(self, messages: List[dict]):
        for message in messages:
            detections_in_db = self.db_client.query(Detection).find_all(
                {"aid": message["aid"]}
            )
            self.clean_detections_from_db(detections_in_db)
            non_detections_in_db = self.db_client.query(NonDetection).find_all(
                {"aid": message["aid"]}
            )
            self.clean_non_detections_from_db(non_detections_in_db)

            message["detections"] = self.unique_detections(
                detections_in_db, message["detections"]
            )
            message["non_detections"] = self.unique_non_detections(
                non_detections_in_db, message["non_detections"]
            )

        return messages

    @staticmethod
    def clean_detections_from_db(detections):
        """Modifies inplace"""
        for detection in detections:
            if "candid" not in detection:  # Compatibility with old DB definitions
                detection["candid"] = detection["_id"]
            detection.pop("_id")

    @staticmethod
    def clean_non_detections_from_db(non_detections):
        """Modifies inplace"""
        for non_detection in non_detections:
            non_detection.pop("_id")
