import unittest
import pytest
import json
from time import sleep
from mongo_scribe.step import MongoScribe
from mongo_scribe.db.models import get_model_collection
from apf.consumers.kafka import KafkaConsumer
from apf.producers.kafka import KafkaProducer

DB_CONFIG = {
    "MONGO": {
        "HOST": "localhost",
        "USERNAME": "mongo",
        "PASSWORD": "mongo",
        "PORT": 27017,
        "DATABASE": "test",
    }
}

CONSUMER_CONFIG = {
    "TOPICS": ["test_topic"],
    "PARAMS": {
        "bootstrap.servers": "localhost:9094",
        "group.id": "command_consumer_1",
        "enable.partition.eof": True,
        "auto.offset.reset": "beginning",
    },
    "NUM_MESSAGES": 2,
}

PRODUCER_CONFIG = {
    "TOPIC": "test_topic",
    "PARAMS": {"bootstrap.servers": "localhost:9094"},
    "SCHEMA": {
        "namespace": "db_operation",
        "type": "record",
        "name": "Command",
        "fields": [
            {"name": "payload", "type": "string"},
        ],
    },
}


@pytest.mark.usefixtures("kafka_service")
@pytest.mark.usefixtures("mongo_service")
class StepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step_config = {
            "DB_CONFIG": DB_CONFIG,
            "STEP_METADATA": {
                "STEP_ID": "scribe",
                "STEP_NAME": "scribe",
                "STEP_VERSION": "test",
                "STEP_COMMENTS": "test ver.",
            },
        }

        konsumer = KafkaConsumer(config=CONSUMER_CONFIG)
        cls.step = MongoScribe(consumer=konsumer, config=cls.step_config)
        cls.producer = KafkaProducer(config=PRODUCER_CONFIG)

    def test_write_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {"field": "some_value"},
            }
        )
        self.producer.produce({"payload": command}, key="insertion_1")
        self.producer.produce({"payload": command}, key="insertion_2")

        self.step.start()
        collection = get_model_collection(
            self.step.db_client.connection, "object"
        )
        result = collection.find({})
        self.assertIsNotNone(result[0])
        self.assertEqual(result[0]["field"], "some_value")
