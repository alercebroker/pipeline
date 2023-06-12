import json
import logging
from datetime import datetime
from typing import List, Union

from apf.consumers import GenericConsumer
from apf.core.step import GenericStep
from apf.producers import GenericProducer

from .strategies.base import BaseStrategy


class StampClassifierStep(GenericStep):
    """AtlasStampClassifierStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    """

    def __init__(
        self,
        consumer: GenericConsumer,
        producer: GenericProducer,
        scribe_producer: GenericProducer,
        strategy: BaseStrategy,
        level=logging.INFO,
    ):
        super().__init__(consumer, level=level)
        self.logger.info("Loading model")
        self.producer = producer
        self.scribe_producer = scribe_producer
        self.strategy = strategy

    def format_output_message(self, predictions: dict) -> List[dict]:
        return [
            {
                "aid": aid,
                "classifications": [
                    {"class_name": cls, "probability": p} for cls, p in probs.items()
                ],
                "classifier_name": self.strategy.name,
                "classifier_version": self.strategy.version,
                "brokerPublishTime": int(datetime.utcnow().timestamp() * 1000),
            }
            for aid, probs in predictions.items()
        ]

    def write_predictions(self, predictions: dict):
        with_metadata = {
            aid: {
                "classifier_name": self.strategy.name,
                "classifier_version": self.strategy.version,
                **probs,
            }
            for aid, probs in predictions.items()
        }

        for aid, probabilities in with_metadata.items():
            data_to_produce = {
                "payload": json.dumps(
                    {
                        "collection": "object",
                        "type": "update_probabilities",
                        "criteria": {"_id": aid},
                        "data": probabilities,
                        "options": {"upsert": True, "set_on_insert": True},
                    }
                )
            }
            self.scribe_producer.produce(data_to_produce)

    def produce(self, output_messages):
        for message in output_messages:
            aid = message["aid"]
            self.producer.produce(message, key=str(aid))

    def add_class_metrics(self, predictions: dict) -> None:
        self.metrics["class"] = {
            aid: max(probs, key=probs.get) for aid, probs in predictions.items()
        }

    def execute(self, messages: Union[List[dict], dict]):
        if isinstance(messages, dict):
            messages = [messages]
        self.logger.info(f"Processing {len(messages)} messages.")

        self.logger.info("Doing inference")
        predictions = self.strategy.get_probabilities(messages)
        self.logger.info(f"Classified {len(predictions)} unique objects")
        if not len(predictions):
            return

        self.logger.info("Inserting/Updating results on database")
        self.write_predictions(predictions)

        self.logger.info("Producing messages")
        output = self.format_output_message(predictions)
        self.produce(output)
        self.add_class_metrics(predictions)
