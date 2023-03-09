import json
import os
import pytest
import unittest
from mongo_scribe.step import MongoScribe
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

    def test_insert_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {"_id": "inserted_id1", "field": "some_value"},
            }
        )
        self.producer.produce({"payload": command}, key="insertion_1")
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {"_id": "inserted_id2", "field": "some_value"},
            }
        )
        self.producer.produce({"payload": command}, key="insertion_2")

        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find_one({"_id": "inserted_id1"})
        self.assertIsNotNone(result)
        self.assertEqual(result["field"], "some_value")

    def test_upsert_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": "upserted_id"},
                "data": {"field": "some_value"},
                "options": {"upsert": True}
            }
        )
        self.producer.produce({"payload": command}, key="insertion_3")
        self.producer.produce({"payload": command}, key="insertion_4")

        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find_one({"_id": "upserted_id"})
        self.assertIsNotNone(result)
        self.assertEqual(result["field"], "some_value")

    def test_insert_probabilities_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert_probabilities",
                "criteria": {"_id": "insert_probabilities_id"},
                "data": {"class1": 0.1, "class2": 0.9, "classifier_name": "classifier", "classifier_version": "1"},
                "options": {"upsert": True}
            }
        )
        self.producer.produce({"payload": command}, key="insertion_5")
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert_probabilities",
                "criteria": {"_id": "insert_probabilities_id"},
                "data": {"class1": 0.5, "class2": 0.3, "classifier_name": "classifier", "classifier_version": "1"},
                "options": {"upsert": True}
            }
        )
        self.producer.produce({"payload": command}, key="insertion_6")
        self.producer.produce({"payload": command}, key="insertion_7")

        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find_one({"_id": "insert_probabilities_id"})
        self.assertIsNotNone(result)
        self.assertIn(
            {
                "classifier_name": "classifier",
                "classifier_version": "1",
                "probability": 0.1,
                "class_name": "class1",
                "ranking": 2
             }, result["probabilities"])
        self.assertIn(
            {
                "classifier_name": "classifier",
                "classifier_version": "1",
                "probability": 0.9,
                "class_name": "class2",
                "ranking": 1
            }, result["probabilities"])

    def test_update_probabilities_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "update_probabilities",
                "criteria": {"_id": "update_probabilities_id"},
                "data": {"class1": 0.1, "class2": 0.9, "classifier_name": "classifier", "classifier_version": "1"},
                "options": {"upsert": True}
            }
        )
        self.producer.produce({"payload": command}, key="insertion_8")
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert_probabilities",
                "criteria": {"_id": "update_probabilities_id"},
                "data": {"class1": 0.5, "class2": 0.3, "classifier_name": "classifier", "classifier_version": "1"},
                "options": {"upsert": True}
            }
        )
        self.producer.produce({"payload": command}, key="insertion_9")
        self.producer.produce({"payload": command}, key="insertion_10")

        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find_one({"_id": "update_probabilities_id"})
        self.assertIsNotNone(result)
        self.assertIn(
            {
                "classifier_name": "classifier",
                "classifier_version": "1",
                "probability": 0.5,
                "class_name": "class1",
                "ranking": 1
            }, result["probabilities"])
        self.assertIn(
            {
                "classifier_name": "classifier",
                "classifier_version": "1",
                "probability": 0.3,
                "class_name": "class2",
                "ranking": 2
            }, result["probabilities"])

    def test_print_into_console(self):
        os.environ["MOCK_DB_COLLECTION"] = "True"
        commands = [
            json.dumps({
                "collection": "object",
                "type": "insert",
                "data": {"field": "some printed value"}
            }),
            json.dumps({
                "collection": "object",
                "type": "update",
                "criteria": {"field": "some printed value"},
                "data": {"field2": "hehe"}
            })
        ]
        self.producer.produce({"payload": commands[0]}, key="insertion")
        self.producer.produce({"payload": commands[1]}, key="update")
        self.step.start()

        os.environ["MOCK_DB_COLLECTION"] = ""
        collection = self.step.db_client.connection.database["object"]
        result = collection.find({"field2": "hehe"})
        self.assertEqual(len(list(result)), 0)
