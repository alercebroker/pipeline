import json
import logging

from apf.core import get_class
from apf.core.step import GenericStep

from .core.strategy import corrector_factory


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

    def execute(self, messages: list[dict]) -> list[dict]:
        self.logger.info(f"Processing {len(messages)} new alerts")
        result = []
        for message in messages:
            detections = [{**message["new_alert"], "has_stamp": True}] + \
                         [{**prv, "has_stamp": False} for prv in message["prv_detections"]]
            strategy = corrector_factory(detections, message["new_alert"]["tid"])
            out_message = {
                "aid": message["aid"],
                "detections": strategy.corrected_message(),
                "non_detections": message["non_detections"]
            }

            result.append(out_message)
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
