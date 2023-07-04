from ..utils.no_class_post_processor import (
    NoClassifiedPostProcessor,
)
from .kafka_parser import KafkaOutput, KafkaParser
from lc_classification.predictors.predictor.predictor_parser import PredictorOutput
from lc_classification.core.parsers.classes.elasticc_mapper import ClassMapper
import pandas as pd
import datetime


class ElasticcParser(KafkaParser):
    def __init__(self):
        super().__init__(ClassMapper)

    def parse(self, model_output: PredictorOutput, **kwargs) -> KafkaOutput[list]:
        # create a hashmap that contains the new info (candid, oid and timestamps)
        detection_extra_info = {}

        messages = kwargs["messages"]
        for message in messages:
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
                    "timestamp"
                ),
                "brokerIngestTimestamp": new_detection["extra_fields"].get(
                    "brokerIngestTimestamp"
                ),
            }
        predictions = model_output.classifications["probabilities"]
        messages = kwargs.get("messages", pd.DataFrame())
        messages = pd.DataFrame().from_records(messages)
        predictions = NoClassifiedPostProcessor(
            messages, predictions
        ).get_modified_classifications()
        predictions["aid"] = predictions.index
        classifier_name = kwargs["classifier_name"]
        classifier_version = kwargs["classifier_version"]
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
                "elasticcPublishTimestamp": detection_extra_info[aid][
                    "elasticcPublishTimestamp"
                ]
                * 1000,
                "brokerIngestTimestamp": detection_extra_info[aid][
                    "brokerIngestTimestamp"
                ]
                * 1000,
                "classifications": output_classification,
                "brokerVersion": classifier_version,
                "classifierName": classifier_name,
                "classifierParams": classifier_version,
                "brokerName": "ALeRCE",
                "brokerPublishTimestamp": int(
                    datetime.datetime.now().timestamp() * 1000
                ),
            }
            output.append(response)
        return KafkaOutput(output)
