import json
from typing import List

from apf.core import get_class
from apf.core.step import GenericStep

from .core.extractor import PreviousCandidatesExtractor


class PrvCandidatesStep(GenericStep):
    """PrvDetectionStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    """

    def __init__(
        self,
        config,
        **step_args,
    ):
        super().__init__(config=config, **step_args)
        producer_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = producer_class(self.config["SCRIBE_PRODUCER_CONFIG"])
        # monitoring
        self.candid_found = False

    def _extract_timestamp(self, message):
        def __extract_timestamp(detection):
            detection.pop("timestamp", None)
            return detection

        dets = message["detections"]
        dets = list(map(__extract_timestamp, dets))
        message["detections"] = dets

        return message

    def pre_produce(self, result: List[dict]):
        self.set_producer_key_field("oid")
        result = list(map(self._extract_timestamp, result))
        return result

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} alerts")
        extractor = PreviousCandidatesExtractor(messages)
        return extractor.extract_all()

    def post_execute(self, result: List[dict]):
        # Produce a message with the non_detections
        for alert in result:
            non_detections = alert["non_detections"]
            self.produce_scribe(non_detections)

        return result

    def produce_scribe(self, non_detections: List[dict]):
        for non_detection in non_detections:
            scribe_data = {
                "collection": "non_detection",
                "type": "update",
                "criteria": {
                    "oid": non_detection["oid"],
                    "fid": non_detection["fid"],
                    "mjd": non_detection["mjd"],
                },
                "data": non_detection,
                "options": {"upsert": True},
            }
            scribe_payload = {"payload": json.dumps(scribe_data)}
            self.scribe_producer.produce(scribe_payload)
