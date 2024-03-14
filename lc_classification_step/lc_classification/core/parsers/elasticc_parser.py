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
            new_detection = [
                det
                for det in message["detections"]
                if det["new"] and det["has_stamp"]
            ]

            if len(new_detection) == 0:
                # case when no new detections
                new_detection = [
                    det for det in message["detections"] if det["has_stamp"]
                ]
                # sort by mjd so the first one is the most recent
                new_detection = sorted(
                    new_detection, key=lambda x: x["mjd"], reverse=True
                )
                if len(new_detection) == 0:
                    raise Exception(
                        "No new detections and no detections with stamps"
                    )

            new_detection = new_detection[0]

            detection_extra_info[new_detection["oid"]] = {
                "candid": new_detection["extra_fields"].get(
                    "alertId", new_detection["candid"]
                ),
                "diaSourceId": new_detection["extra_fields"].get(
                    "diaSourceId", new_detection["candid"]
                ),
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
        predictions["oid"] = predictions.index
        for class_name in self.ClassMapper.get_class_names():
            if class_name not in predictions.columns:
                predictions[class_name] = 0.0
        classifications = predictions.to_dict(orient="records")
        output = []
        for classification in classifications:
            oid = classification.pop("oid")
            if "classifier_name" in classification:
                classification.pop("classifier_name")

            output_classification = [
                {
                    "classId": ClassMapper.get_class_value(predicted_class),
                    "probability": prob,
                }
                for predicted_class, prob in classification.items()
            ]
            print(detection_extra_info)
            response = {
                "alertId": int(detection_extra_info[oid]["candid"]),
                "diaSourceId": int(detection_extra_info[oid]["diaSourceId"]),
                "elasticcPublishTimestamp": int(
                    detection_extra_info[oid]["elasticcPublishTimestamp"]
                ),
                "brokerIngestTimestamp": int(
                    detection_extra_info[oid]["brokerIngestTimestamp"]
                ),
                "classifications": output_classification,
                "brokerVersion": classifier_version,
                "classifierName": classifier_name,
                "classifierParams": classifier_version,
                "brokerName": "ALeRCE",
            }
            output.append(response)
        return KafkaOutput(output)
