import json
import logging
from functools import reduce

from apf.core import get_class
from apf.core.step import GenericStep

from .core.strategy import ZTFStrategy


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

    def pre_produce(self, result: list[dict]):
        self.set_producer_key_field("aid")
        return result

    @staticmethod
    def _get_detections(message: dict) -> list:
        def has_stamp(detection, stamp=True):
            return {**detection, "has_stamp": stamp}

        return [has_stamp(message["new_alert"])] + [has_stamp(prv, False) for prv in message["prv_detections"]]

    def pre_execute(self, messages: list[dict]) -> dict:
        detections, non_detections = [], []
        for msg in messages:
            detections.extend(self._get_detections(msg))
            non_detections.extend(msg["non_detections"])
        return {"detections": detections, "non_detections": non_detections}

    def execute(self, message: dict) -> list[dict]:
        self.logger.info(f"Processing {len(message)} new alerts")
        corrector = ZTFStrategy(**message)
        output = corrector.corrected_message()

        return result

    def post_execute(self, result: list[dict]):
        for message in result:
            self.produce_scribe(message["detections"])
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
