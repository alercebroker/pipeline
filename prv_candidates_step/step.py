import json
import logging
from typing import List

from apf.core import get_class
from apf.core.step import GenericStep

from .core.candidates.process_prv_candidates import process_prv_candidates
from .core.utils.remove_keys import remove_keys_from_dictionary
from .core import ZTFPreviousDetectionsParser, ZTFNonDetectionsParser


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
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)
        producer_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = producer_class(self.config["SCRIBE_PRODUCER_CONFIG"])

    def pre_produce(self, result: List[dict]):
        self.set_producer_key_field("aid")
        return result

    def _get_parsers(self, survey: str):
        if survey.lower().startswith("ztf"):
            return ZTFPreviousDetectionsParser(), ZTFNonDetectionsParser()

        return None, None

    def execute(self, messages):
        self.logger.info("Processing %s alerts", str(len(messages)))
        prv_detections, non_detections = process_prv_candidates(messages)
        output = []
        for index, alert in enumerate(messages):
            prv_detections_parser, non_detections_parser = self._get_parsers(
                alert["tid"]
            )

            stampless_alert = remove_keys_from_dictionary(alert, ["stamps"])
            stampless_alert["has_stamp"] = True

            if prv_detections_parser is None:
                parsed_prv_detections = []
                parsed_non_detections = []

            else:
                parsed_prv_detections = prv_detections_parser.parse(
                    prv_detections[index], alert["oid"], alert["aid"], alert["candid"]
                )
                parsed_prv_detections = [
                    remove_keys_from_dictionary(prv, ["stamps"])
                    for prv in parsed_prv_detections
                ]
                parsed_non_detections = non_detections_parser.parse(
                    non_detections[index], alert["aid"], alert["tid"], alert["oid"]
                )
                # move fields to extra_fields before removing them
                stampless_alert["extra_fields"] = remove_keys_from_dictionary(
                    stampless_alert["extra_fields"], ["prv_candidates"]
                )
                
            if stampless_alert["tid"].lower().startswith("ztf"):
                stampless_alert["extra_fields"]["parent_candid"] = None

            output.append(
                {
                    "aid": alert["aid"],
                    "detections": [stampless_alert, *parsed_prv_detections],
                    "non_detections": parsed_non_detections,
                }
            )
        return output

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
