import json
import logging

import pandas as pd
from apf.core import get_class
from apf.core.step import GenericStep

from .core import Corrector


class CorrectionStep(GenericStep):
    """Step that applies magnitude correction to new alert and previous candidates.

    The correction refers to passing from the magnitude measured from the flux in the difference
    stamp to the actual apparent magnitude. This requires a reference magnitude to work.
    """

    def __init__(
        self,
        config,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)
        cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])
        self.set_producer_key_field("aid")

    @classmethod
    def pre_produce(cls, result: dict):
        detections = pd.DataFrame(result["detections"]).groupby("aid")
        non_detections = pd.DataFrame(result["non_detections"]).groupby("aid")
        output = []
        for aid, dets in detections:
            try:
                nd = non_detections.get_group(aid).to_dict("records")
            except KeyError:
                nd = []
            output.append({"aid": aid, "detections": dets.to_dict("records"), "non_detections": nd})
        return output

    @staticmethod
    def _get_detections(message: dict) -> list:
        def has_stamp(detection, stamp=True):
            return {**detection, "has_stamp": stamp}
        return [has_stamp(message["new_alert"])] + [has_stamp(prv, False) for prv in message["prv_detections"]]

    @classmethod
    def pre_execute(cls, messages: list[dict]) -> dict:
        detections, non_detections = [], []
        for msg in messages:
            detections.extend(cls._get_detections(msg))
            non_detections.extend(msg["non_detections"])
        return {"detections": detections, "non_detections": non_detections}

    @classmethod
    def execute(cls, message: dict) -> dict:
        corrector = Corrector(message["detections"])
        detections = corrector.corrected_records()
        non_detections = pd.DataFrame(message["non_detections"])
        non_detections = non_detections.drop_duplicates(["oid", "fid", "mjd"])
        return {"detections": detections, "non_detections": non_detections}

    def post_execute(self, result: dict):
        self.produce_scribe(result["detections"])
        return result

    def produce_scribe(self, detections: list[dict]):
        for detection in detections:
            detection = detection.copy()  # Prevent modification for next step
            candid = detection.pop("candid")
            set_on_insert = not detection["has_stamp"]
            scribe_data = {
                "collection": "detection",
                "type": "update",
                "criteria": {"_id": candid},
                "data": detection,
                "options": {"upsert": True, "set_on_insert": set_on_insert},
            }
            scribe_payload = {"payload": json.dumps(scribe_data)}
            self.scribe_producer.produce(scribe_payload)
