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
        self.db_client.connect(config["DB_CONFIG"])

    @staticmethod
    def unique_detections(old_detections, new_detections):
        """Return only non-duplicate detections (based on candid).

        Will always keep detections with stamps over ones without. Otherwise, it will keep the
        ones in `old_detections` over those in `new_detections`.
        """
        new_candids_with_stamp = [
            det["candid"] for det in new_detections if det["has_stamp"]
        ]

        old_detections = [
            det for det in old_detections if det["candid"] not in new_candids_with_stamp
        ]
        candids = [det["candid"] for det in old_detections]

        new_detections = [det for det in new_detections if det["candid"] not in candids]
        return old_detections + new_detections

    @staticmethod
    def unique_non_detections(old_non_detections, new_non_detections):
        """Return only non-duplicate non-detections (based on `oid`, `fid` and `mjd`).

        Keeps the ones in `old_non_detections` over those in `new_non_detections`.
        """

        def create_id(detection):
            return {k: v for k, v in detection.items() if k in ["oid", "fid", "mjd"]}

        ids = [create_id(det) for det in old_non_detections]

        new_non_detections = [
            det for det in new_non_detections if create_id(det) not in ids
        ]
        return old_non_detections + new_non_detections

    @classmethod
    def pre_execute(cls, messages: List[dict]) -> List[dict]:
        """If multiple AIDs are in the same batch create a single message with all of them"""
        aids, output = {}, []
        for message in messages:
            if message["aid"] in aids:
                idx = aids[message["aid"]]
                output[idx]["detections"] = cls.unique_detections(
                    output[idx]["detections"], message["detections"]
                )
                output[idx]["non_detections"] = cls.unique_non_detections(
                    output[idx]["non_detections"], message["non_detections"]
                )
            else:
                output.append(message)
                aids[message["aid"]] = len(output) - 1
        return output

    def execute(self, messages: List[dict]) -> List[dict]:
        """Queries the database for all detections and non-detections for each AID and removes duplicates"""
        for message in messages:
            detections_in_db = self.db_client.query(Detection).find_all(
                {"aid": message["aid"]}, paginate=False
            )
            self.clean_detections_from_db(detections_in_db)
            non_detections_in_db = self.db_client.query(NonDetection).find_all(
                {"aid": message["aid"]}, paginate=False
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
        """Removes field `_id` from detections coming from the database.

        For compatibility with old database, if the field `candid` is present in the detections, this
        will simply remove the `_id` field. If `candid` is not present, assumes that `_id` is the candid.
        In that case, it will remove `_id` and set it as the `candid` field.

        **Modifies detections inplace**
        """
        for detection in detections:
            if "candid" not in detection:  # Compatibility with old DB definitions
                detection["candid"] = detection["_id"]
            detection.pop("_id")

    @staticmethod
    def clean_non_detections_from_db(non_detections):
        """Removes field `_id` from non detections coming from the database.

        **Modifies inplace**
        """
        for non_detection in non_detections:
            non_detection.pop("_id")
