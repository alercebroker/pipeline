from ..utils.no_class_post_processor import (
    NoClassifiedPostProcessor,
)
from .kafka_parser import KafkaOutput, KafkaParser
from lc_classification.core.parsers.classes.elasticc_mapper import ClassMapper
from alerce_classifiers.base.dto import OutputDTO
import pandas as pd
from typing import List


class ElasticcParser(KafkaParser):
    def __init__(self):
        super().__init__(ClassMapper)

    def parse(
        self,
        model_output: OutputDTO,
        *,
        messages: List[dict],
        classifier_name: str,
        classifier_version: str,
        **kwargs,
    ) -> KafkaOutput[list]:
        # create a hashmap that contains the new info (candid, oid and timestamps)
        detection_extra_info = {}

        for message in messages:
            # for iteration con continue deberia ser mas simple
            new_detection = [
                det for det in message["detections"] if det["new"] and det["has_stamp"]
            ]

            if len(new_detection) == 0:
                continue

            new_detection = new_detection[0]

            detection_extra_info[new_detection["aid"]] = {
                "candid": new_detection["candid"],
                "oid": new_detection["oid"],
                "elasticcPublishTimestamp": new_detection["extra_fields"].get(
                    "surveyPublishTimestamp"
                ),
                "brokerIngestTimestamp": new_detection["extra_fields"].get(
                    "brokerIngestTimestamp"
                ),
            }
        predictions = model_output.probabilities
        messages = pd.DataFrame().from_records(messages)
        predictions = NoClassifiedPostProcessor(
            messages, predictions
        ).get_modified_classifications()
        predictions["aid"] = predictions.index
        for class_name in self.ClassMapper.get_class_names():
            if class_name not in predictions.columns:
                predictions[class_name] = 0.0
        classifications = predictions.to_dict(orient="records")
        output = []
        for classification in classifications:
            aid = classification.pop("aid")
            if "classifier_name" in classification:
                classification.pop("classifier_name")

            output_classification = [
                {
                    "classId": ClassMapper.get_class_value(predicted_class),
                    "probability": prob,
                }
                for predicted_class, prob in classification.items()
            ]
            response = {
                "alertId": int(detection_extra_info[aid]["candid"]),
                "diaSourceId": int(detection_extra_info[aid]["oid"]),
                "elasticcPublishTimestamp": int(
                    detection_extra_info[aid]["elasticcPublishTimestamp"]
                ),
                "brokerIngestTimestamp": int(
                    detection_extra_info[aid]["brokerIngestTimestamp"]
                ),
                "classifications": output_classification,
                "brokerVersion": classifier_version,
                "classifierName": classifier_name,
                "classifierParams": classifier_version,
                "brokerName": "ALeRCE",
            }
            output.append(response)
        return KafkaOutput(output)
