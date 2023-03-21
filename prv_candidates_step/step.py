from typing import List
from apf.core.step import GenericStep
from prv_candidates_step.core.candidates.process_prv_candidates import (
    process_prv_candidates,
)
from prv_candidates_step.core.utils.remove_keys import remove_keys_from_dictionary
from prv_candidates_step.core.strategy.ztf_strategy import ZTFPrvCandidatesStrategy
from prv_candidates_step.core.processor.processor import Processor
from prv_candidates_step.core import ZTFPreviousDetectionsParser, ZTFNonDetectionsParser
import logging
import json
from apf.core import get_class


class PrvCandidatesStep(GenericStep):
    """PrvDetectionStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(
        self,
        config,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)
        self.prv_candidates_processor = Processor(
            ZTFPrvCandidatesStrategy()
        )  # initial strategy (can change)
        producer_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = producer_class(self.config["SCRIBE_PRODUCER_CONFIG"])

    def pre_produce(self, result: List[dict]):
        self.set_producer_key_field("aid")
        return result

    def execute(self, messages):
        self.logger.info("Processing %s alerts", str(len(messages)))
        prv_detections, non_detections = process_prv_candidates(
            self.prv_candidates_processor, messages
        )
        output = []
        for index, alert in enumerate(messages):
            ztf_prv_detections_parser = ZTFPreviousDetectionsParser()
            ztf_non_detections_parser = ZTFNonDetectionsParser()
            parsed_prv_detections = ztf_prv_detections_parser.parse(
                prv_detections[index], alert["oid"], alert["aid"], alert["candid"]
            )
            parsed_prv_detections = [
                remove_keys_from_dictionary(prv, ["stamps", "rfid", "rb", "rbversion"])
                for prv in parsed_prv_detections
            ]
            parsed_non_detections = ztf_non_detections_parser.parse(
                non_detections[index], alert["aid"], alert["tid"], alert["oid"]
            )
            stampless_alert = remove_keys_from_dictionary(
                alert, ["stamps", "rfid", "rb", "rbversion"]
            )
            stampless_alert["extra_fields"] = remove_keys_from_dictionary(
                stampless_alert["extra_fields"], ["prv_candidates"]
            )
            output.append(
                {
                    "aid": alert["aid"],
                    "new_alert": stampless_alert,
                    "prv_detections": parsed_prv_detections,
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
                "criteria": {"_id": self._generate_id(non_detection)},
                "data": non_detection,
                "options": {"upsert": True},
            }
            scribe_payload = {"payload": json.dumps(scribe_data)}
            self.scribe_producer.produce(scribe_payload)

    @staticmethod
    def _generate_id(non_detection):
        return hash((non_detection["oid"], non_detection["fid"], non_detection["mjd"]))
