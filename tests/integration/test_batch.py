import json
import os
import pytest
import unittest
from mongo_scribe.step import MongoScribe
from apf.consumers.kafka import KafkaConsumer
from apf.producers.kafka import KafkaProducer
from _generator import CommandGenerator

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
        "bootstrap.servers": "localhost:9092",
        "group.id": "command_consumer_1",
        "enable.partition.eof": True,
        "auto.offset.reset": "beginning",
    },
    "NUM_MESSAGES": 2,
}

PRODUCER_CONFIG = {
    "TOPIC": "test_topic",
    "PARAMS": {"bootstrap.servers": "localhost:9092"},
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

        generator = CommandGenerator()
        cls.commands = [ generator.generate_random_command() for i in range(50) ]
        cls.generator = generator

    def test_bulk(self):
        for i, command in enumerate(self.commands):
            self.producer.produce(command, key=f"key_{i}")

        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find()
        self.assertIsNotNone(result)
        print(list(result))


