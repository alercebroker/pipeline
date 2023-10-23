import json
import os
import pytest
import unittest
from mongo_scribe.step import MongoScribe
from apf.producers.kafka import KafkaProducer
from db_plugins.db.mongo._connection import MongoConnection

DB_CONFIG = {
    "MONGO": {
        "host": "localhost",
        "username": "mongo",
        "password": "mongo",
        "port": 27017,
        "database": "test",
    }
}

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "TOPICS": ["test_topic"],
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "command_consumer_2",
        "enable.partition.eof": True,
        "auto.offset.reset": "beginning",
    },
    "NUM_MESSAGES": 2,
    "TIMEOUT": 10,
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


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
class MongoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = MongoConnection(DB_CONFIG["MONGO"])
        step_config = {
            "DB_CONFIG": DB_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "STEP_METADATA": {
                "STEP_ID": "scribe",
                "STEP_NAME": "scribe",
                "STEP_VERSION": "test",
                "STEP_COMMENTS": "test ver.",
            },
        }
        cls.db.create_db()
        cls.step = MongoScribe(config=step_config)
        cls.producer = KafkaProducer(config=PRODUCER_CONFIG)

    def tearDown(self):
        object_collection = self.db.database["object"]
        object_collection.delete_many({})
        object_detection = self.db.database["detection"]
        object_detection.delete_many({})

    def test_insert_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {"_id": "inserted_id1", "field": "some_value"},
            }
        )
        self.producer.produce({"payload": command})
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {"_id": "inserted_id2", "field": "some_value"},
            }
        )
        self.producer.produce({"payload": command})
        self.producer.producer.flush(1)
        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find_one({"_id": "inserted_id1"})
        assert result is not None
        assert result["field"] == "some_value"

    def test_insert_into_multiple_collections(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {"_id": "inserted_id10", "field": "some_value"},
            }
        )
        self.producer.produce({"payload": command})
        command = json.dumps(
            {
                "collection": "detection",
                "type": "insert",
                "data": {"_id": "inserted_id2", "field": "some_value2"},
            }
        )
        self.producer.produce({"payload": command})
        self.producer.producer.flush(1)
        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        detection_coll = self.step.db_client.connection.database["detection"]
        result = collection.find_one({"_id": "inserted_id10"})
        assert result is not None
        assert result["field"] == "some_value"
        result_detection = detection_coll.find_one({"_id": "inserted_id2"})
        assert result_detection is not None
        assert result_detection["field"] == "some_value2"

    def test_upsert_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": "upserted_id"},
                "data": {"field": "some_value"},
                "options": {"upsert": True},
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command})
        self.producer.producer.flush(1)
        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find_one({"_id": "upserted_id"})
        assert result is not None
        assert result["field"] == "some_value"

    def test_upsert_only_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": "upserted_only_id"},
                "data": {"field": "some_value"},
                "options": {"upsert": True, "set_on_insert": True},
            }
        )
        self.producer.produce({"payload": command})
        command = json.dumps(
            {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": "upserted_only_id"},
                "data": {"field": "other_value"},
                "options": {"upsert": True, "set_on_insert": True},
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command})
        self.producer.producer.flush(1)
        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find_one({"_id": "upserted_only_id"})
        assert result is not None
        assert result["field"] == "some_value"

    def test_insert_probabilities_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "update_probabilities",
                "criteria": {"_id": "insert_probabilities_id"},
                "data": {
                    "class1": 0.1,
                    "class2": 0.9,
                    "classifier_name": "classifier",
                    "classifier_version": "1",
                },
                "options": {"upsert": True},
            }
        )
        self.producer.produce({"payload": command})
        self.producer.producer.flush(1)
        self.producer.produce({"payload": command})
        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find_one({"_id": "insert_probabilities_id"})
        assert result is not None
        assert len(result["probabilities"]) == 1
        probabilities = result["probabilities"][0]
        assert probabilities["classifier_name"] == "classifier"
        assert probabilities["version"] == "1"
        assert probabilities["class_rank_1"] == "class2"
        assert probabilities["probability_rank_1"] == 0.9
        assert len(probabilities["values"]) == 2

    def test_update_probabilities_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "update_probabilities",
                "criteria": {"_id": "update_probabilities_id"},
                "data": {
                    "class1": 0.1,
                    "class2": 0.9,
                    "classifier_name": "classifier",
                    "classifier_version": "1",
                },
                "options": {"upsert": True},
            }
        )
        self.producer.produce({"payload": command})
        command = json.dumps(
            {
                "collection": "object",
                "type": "update_probabilities",
                "criteria": {"_id": "update_probabilities_id"},
                "data": {
                    "class1": 0.5,
                    "class2": 0.3,
                    "classifier_name": "classifier",
                    "classifier_version": "1",
                },
                "options": {"upsert": True},
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command})
        self.producer.producer.flush(1)
        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find_one({"_id": "update_probabilities_id"})
        assert result is not None
        assert len(result["probabilities"]) == 1
        probabilities = result["probabilities"][0]
        assert probabilities["classifier_name"] == "classifier" 
        assert probabilities["version"] == "1"
        assert probabilities["class_rank_1"] == "class1"
        assert probabilities["probability_rank_1"] == 0.5

    def test_update_features_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "update_features",
                "criteria": {"_id": "update_features_id"},
                "data": {
                    "features_version": "v1",
                    "features_group": "elasticc",
                    "features": [
                        {"name": "feat1", "value": 123, "fid": "g"},
                        {"name": "feat2", "value": 456, "fid": "Y"},
                    ],
                },
                "options": {"upsert": True},
            }
        )
        self.producer.produce({"payload": command})
        command = json.dumps(
            {
                "collection": "object",
                "type": "update_features",
                "criteria": {"_id": "update_features_id"},
                "data": {
                    "features_version": "v1",
                    "features_group": "elasticc",
                    "features": [
                        {"name": "feat1", "value": 741, "fid": "g"},
                        {"name": "feat2", "value": 369, "fid": "Y"},
                    ],
                },
                "options": {"upsert": True},
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command})

        command = json.dumps(
            {
                "collection": "object",
                "type": "update_features",
                "criteria": {"_id": "update_features_id"},
                "data": {
                    "features_version": "v1",
                    "features_group": "elasticc",
                    "features": [
                        {"name": "feat1", "value": 741, "fid": "g"},
                        {"name": "feat2", "value": 369, "fid": "Y"},
                    ],
                },
                "options": {"upsert": True},
            }
        )
        self.producer.produce({"payload": command})
        self.producer.producer.flush(1)
        self.step.start()
        collection = self.step.db_client.connection.database["object"]
        result = collection.find_one({"_id": "update_features_id"})
        assert result is not None
        assert len(result["features"]) == 1
        features = result["features"][0]
        assert features["survey"] == "elasticc"
        assert features["version"] == "v1"
        assert {
            "name": "feat1",
            "value": 741,
            "fid": "g",
        } in features["features"]

    def test_update_non_detections_into_database(self):
        command = json.dumps(
            {
                "collection": "non_detection",
                "type": "update",
                "criteria": {
                    "oid": "ZTF12345",
                    "aid": "ALX534311",
                    "fid": "g",
                    "mjd": 55500,
                },
                "data": {
                    "diffmaglim": 21.11
                },
                "options": {"upsert": True},
            }
        )
        self.producer.produce({"payload": command})
        command = json.dumps(
            {
                "collection": "non_detection",
                "type": "update",
                "criteria": {
                    "oid": "ZTF12345",
                    "aid": "ALX534311",
                    "fid": "g",
                    "mjd": 55500,
                },
                "data": {
                    "diffmaglim": 33.33
                },
                "options": {"upsert": True},
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command})

        command = json.dumps(
            {
                "collection": "non_detection",
                "type": "update",
                "criteria": {
                    "oid": "ZTF12345",
                    "aid": "ALX534311",
                    "fid": "g",
                    "mjd": 55500,
                },
                "data": {
                    "diffmaglim": 11.11
                },
                "options": {"upsert": True},
            }
        )
        self.producer.produce({"payload": command})
        self.producer.producer.flush(1)
        self.step.start()
        collection = self.step.db_client.connection.database["non_detection"]
        result = collection.find_one({"aid": "ALX534311"})
        print(result)

    def test_print_into_console(self):
        os.environ["MOCK_DB_COLLECTION"] = "True"
        commands = [
            json.dumps(
                {
                    "collection": "object",
                    "type": "insert",
                    "data": {"_id": "some printed value"},
                }
            ),
            json.dumps(
                {
                    "collection": "object",
                    "type": "update",
                    "criteria": {"_id": "some printed value"},
                    "data": {"field2": "hehe"},
                }
            ),
        ]
        self.producer.produce({"payload": commands[0]})
        self.producer.produce({"payload": commands[1]})
        self.producer.producer.flush(1)
        self.step.start()

        os.environ["MOCK_DB_COLLECTION"] = ""
        collection = self.step.db_client.connection.database["object"]
        result = collection.find({"_id": "some printed value"})
        assert len(list(result)) == 0
