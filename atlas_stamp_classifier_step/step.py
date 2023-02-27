import json
import logging
from datetime import datetime
from typing import List, Union

from apf.consumers import GenericConsumer
from apf.core.step import GenericStep
from apf.producers import GenericProducer

from .strategies.base import BaseStrategy


class AtlasStampClassifierStep(GenericStep):
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
        config: dict,
        strategy: BaseStrategy,
        level=logging.INFO,
    ):
        super().__init__(consumer, config=config, level=level)
        self.logger.info("Loading model")
        self.strategy = strategy
        self.producer = producer
        self.scribe_producer = scribe_producer

    def format_output_message(self, predictions: dict) -> List[dict]:
        return [
            {
                "aid": aid,
                "classifications": [
                    {
                        k: v
                        for k, v in probability.items()
                        if k not in ("classifier_version", "ranking")
                    }
                    for probability in probabilities
                ],
                "model_version": self.strategy.version,
                "brokerPublishTime": int(datetime.utcnow().timestamp() * 1000),
            }
            for aid, probabilities in predictions.items()
        ]

    def write_predictions(self, predictions: dict):
        for aid, probabilities in predictions.items():
            data_to_produce = {
                "payload": json.dumps(
                    {
                        "collection": "object",
                        "type": "update-probabilities",
                        "criteria": {"aid": aid},
                        "data": {"probabilities": probabilities},
                    }
                )
            }
            self.scribe_producer.produce(data_to_produce)

    def produce(self, output_messages):
        for message in output_messages:
            aid = message["aid"]
            self.producer.produce(message, key=str(aid))

    def add_class_metrics(self, predictions: dict) -> None:
        self.metrics["class"] = [
            probability["class_name"]
            for probabilities in predictions.values()
            for probability in probabilities
            if probability["ranking"] == 1
        ]

    def execute(self, messages: Union[List[dict], dict]):
        if isinstance(messages, dict):
            messages = [messages]
        self.logger.info(f"Processing {len(messages)} messages.")
        self.logger.info("Getting batch alert data")

        self.logger.info("Doing inference")
        predictions = self.strategy.get_probabilities(messages)
        if not len(predictions):
            self.logger.info("No output to write")
            return

        self.logger.info("Inserting/Updating results on database")
        self.write_predictions(predictions)  # should predictions be in normalized form?

        self.logger.info("Producing messages")
        output = self.format_output_message(predictions)
        self.produce(output)
        self.add_class_metrics(predictions)
