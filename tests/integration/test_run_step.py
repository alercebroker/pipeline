import logging
import pytest
import unittest

from apf.consumers import KafkaConsumer
from sorting_hat_step import SortingHatStep
from schema import SCHEMA
from tests.unittest.data.batch import generate_alerts_batch

DB_CONFIG = {
    "HOST": "localhost",
    "USER": "test_user",
    "PASSWORD": "test_password",
    "PORT": 27017,
    "DATABASE": "test_db",
}

PRODUCER_CONFIG = {
    "TOPIC": "sorting_hat_stream",
    "PARAMS": {
        "bootstrap.servers": "localhost:9094",
    },
    "SCHEMA": SCHEMA
}

CONSUMER_CONFIG = {
    "PARAMS": {
        "bootstrap.servers":  "localhost:9094",
        "group.id": "sorting_hat_consumer",
        "auto.offset.reset": "beginning",
        'max.poll.interval.ms': 3600000
    },
    "consume.timeout": 10,
    "consume.messages": 10,
    "TOPICS": ["topic_test"]
}


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
class MongoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        consumer = KafkaConsumer(config=CONSUMER_CONFIG)
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
            consumer,
            config=cls.step_config,
            level=logging.DEBUG
        )

    def test_execute(self):
        batch = generate_alerts_batch(100, nearest=10)  # generate 110 alerts where 10 alerts are near of another alerts
        self.step.execute(batch)
