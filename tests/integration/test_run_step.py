import logging
from typing import List
from apf.core.step import GenericConsumer
from apf.producers import KafkaProducer
from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import MongoDatabaseCreator
import pytest
import unittest

from apf.consumers import KafkaConsumer
from sorting_hat_step import SortingHatStep
from schema import SCHEMA
from tests.unittest.data.batch import generate_alerts_batch

DB_CONFIG = {
    "HOST": "localhost",
    "USERNAME": "test_user",
    "PASSWORD": "test_password",
    "PORT": 27017,
    "DATABASE": "test_db",
    "AUTH_SOURCE": "test_db",
}

PRODUCER_CONFIG = {
    "TOPIC": "sorting_hat_stream",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": SCHEMA,
}

CONSUMER_CONFIG = {
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "sorting_hat_consumer",
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
        "enable.partition.eof": True,
    },
    "consume.timeout": 10,
    "consume.messages": 1,
    "TOPICS": ["sorting_hat_stream"],
}


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
class MongoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        consumer = KafkaConsumer(config=CONSUMER_CONFIG)
        producer = KafkaProducer(config=PRODUCER_CONFIG)
        database = new_DBConnection(MongoDatabaseCreator)
        cls.step_config = {
            "DB_CONFIG": DB_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "STEP_METADATA": {
                "STEP_ID": "ingestion",
                "STEP_NAME": "ingestion",
                "STEP_VERSION": "test",
                "STEP_COMMENTS": "developing and testing it",
            },
        }
        cls.step = SortingHatStep(
            consumer, cls.step_config, producer, database, level=logging.DEBUG
        )

    def test_execute(self):
        batch = generate_alerts_batch(
            100, nearest=10
        )  # generate 110 alerts where 10 alerts are near of another alerts
        self.step.execute(batch)
        messages = self.consume_messages()
        for message in messages:
            # TODO add other assertions
            self.assert_message_stamps(message)

    def consume_messages(self) -> List[dict]:
        config = CONSUMER_CONFIG.copy()
        config["PARAMS"]["group.id"] = "assert"
        consumer = KafkaConsumer(config)
        messages = []
        for message in consumer.consume():
            messages.append(message)
        return messages

    def assert_message_stamps(self, message: dict):
        print(message)
        assert message["stamps"]["science"] == b"science"
        if message["tid"] == "ZTF":
            assert message["stamps"]["template"] == b"template"
        if message["tid"] == "ATLAS":
            assert message["stamps"]["template"] == None
        assert message["stamps"]["difference"] == b"difference"
