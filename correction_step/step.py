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

    def pre_produce(self, result: tuple):
        self.set_producer_key_field("aid")
        output = []
        for index, alert in enumerate(result[0]):
            output.append(
                {
                    "aid": alert["aid"],
                    "detections": result[1][index],
                    "non_detections": alert["non_detections"],
                }
            )
        return output

    def execute(self, messages: list[dict]) -> list[dict]:
        self.logger.info(f"Processing {len(messages)} new alerts")
        result = []
        for message in messages:
            tid = message["new_alert"]["tid"]
            detections = message["prv_detections"] + [message["new_alert"]]

            strategy = corrector_factory(detections, tid)
            detections = strategy.corrected_message()

            result.append(
                {"aid": message["aid"], "detections": detections, "non_detections": message["non_detections"]}
            )
        return result

    def post_execute(self, result: list[dict]):
        for message in result:
            self.produce_scribe(message["detections"])
        return result

    def produce_scribe(self, detections: list[dict]):
        for detection in detections:
            scribe_data = {
                "collection": "detection",
                "type": "update",
                "criteria": {"_id": detection["candid"]},
                "data": detection,
                "options": {"upsert": True},
            }
            scribe_payload = {"payload": json.dumps(scribe_data)}
            self.scribe_producer.produce(scribe_payload)
