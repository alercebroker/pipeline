import json
import logging
from datetime import datetime
from typing import List, Union

from apf.consumers import GenericConsumer
from apf.core.step import GenericStep
from apf.producers import GenericProducer
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.mongo.models import Object

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
        db_connection: MongoConnection,
        level=logging.INFO,
    ):
        super().__init__(consumer, level=level)
        self.logger.info("Loading model")
        self.producer = producer
        self.scribe_producer = scribe_producer
        self.strategy = strategy
        self.db_connection = db_connection

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
                        "type": "insert_probabilities",
                        "criteria": {"aid": aid},
                        "data": probabilities,
                        "options": {"upsert": True}
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

    def _remove_objects_in_database(self, messages: List[dict]):
        def exists(obj):
            try:
                return any(
                    p["classifier_name"] == self.strategy.name
                    for p in obj["probabilities"]
                )
            except KeyError:  # In case the object doesn't have a probabilities field
                return False

        aids = [msg["aid"] for msg in messages]
        objects = self.db_connection.query(Object).find_all(
            {"_id": {"$in": aids}}, paginate=False
        )

        objects_in_db = [obj["_id"] for obj in objects if exists(obj)]
        return [msg for msg in messages if msg["aid"] not in objects_in_db]

    def execute(self, messages: Union[List[dict], dict]):
        if isinstance(messages, dict):
            messages = [messages]
        self.logger.info(f"Processing {len(messages)} messages.")

        self.logger.info("Removing messages from alredy classified objects")
        messages = self._remove_objects_in_database(messages)
        self.logger.info(f"Processing {len(messages)} messages from new objects")
        if not len(messages):
            return

        self.logger.info("Doing inference")
        predictions = self.strategy.get_probabilities(messages)
        self.logger.info(f"Classified {len(predictions)} unique objects")
        if not len(predictions):
            return

        self.logger.info("Inserting/Updating results on database")
        self.write_predictions(predictions)  # should predictions be in normalized form?

        self.logger.info("Producing messages")
        output = self.format_output_message(predictions)
        self.produce(output)
        self.add_class_metrics(predictions)
