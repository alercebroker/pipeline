import json
import os
import pytest
from mongo_scribe.step import MongoScribe
from apf.producers.kafka import KafkaProducer
from db_plugins.db.mongo._connection import MongoConnection

DB_CONFIG = {
    "MONGO": {
        "host": "localhost",
        "username": "",
        "password": "",
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
    "SCHEMA_PATH": os.path.join(
        os.path.dirname(__file__), "producer_schema.avsc"
    ),
}

db = None
step = None
producer = None
step_config = {
    "DB_CONFIG": DB_CONFIG,
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
}


@pytest.fixture
def setUp(mongo_service, kafka_service):
    global db, step, producer
    if db is None:
        db = MongoConnection(DB_CONFIG["MONGO"])
        db.create_db()

    DB_CONFIG["MONGO"]["database"] = db.database.name
    if step is None:
        step = MongoScribe(config=step_config)
        producer = KafkaProducer(config=PRODUCER_CONFIG)
    yield step, producer
    collection = db.database["object"]
    collection.delete_many({})
    collection = db.database["detection"]
    collection.delete_many({})
    collection = db.database["non_detection"]
    collection.delete_many({})


def test_insert_into_database(setUp):
    step, producer = setUp
    command = json.dumps(
        {
            "collection": "object",
            "type": "insert",
            "data": {"_id": "inserted_id1", "field": "some_value"},
        }
    )
    producer.produce({"payload": command})
    command = json.dumps(
        {
            "collection": "object",
            "type": "insert",
            "data": {"_id": "inserted_id2", "field": "some_value"},
        }
    )
    producer.produce({"payload": command})
    producer.producer.flush()
    step.start()
    collection = step.db_client.connection.database["object"]
    result = collection.find_one({"_id": "inserted_id1"})
    assert result is not None
    assert result["field"] == "some_value"


def test_insert_into_multiple_collections(setUp):
    step, producer = setUp
    command = json.dumps(
        {
            "collection": "object",
            "type": "insert",
            "data": {"_id": "inserted_id10", "field": "some_value"},
        }
    )
    producer.produce({"payload": command})
    command = json.dumps(
        {
            "collection": "detection",
            "type": "insert",
            "data": {"_id": "inserted_id2", "field": "some_value2"},
        }
    )
    producer.produce({"payload": command})
    producer.producer.flush(1)
    step.start()
    collection = step.db_client.connection.database["object"]
    detection_coll = step.db_client.connection.database["detection"]
    result = collection.find_one({"_id": "inserted_id10"})
    assert result is not None
    assert result["field"] == "some_value"
    result_detection = detection_coll.find_one({"_id": "inserted_id2"})
    assert result_detection is not None
    assert result_detection["field"] == "some_value2"


def test_upsert_into_database_with_existing_document(setUp):
    step, producer = setUp
    command = json.dumps(
        {
            "collection": "object",
            "type": "update",
            "criteria": {"_id": "upserted_id"},
            "data": {"field": "value1"},
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    command = json.dumps(
        {
            "collection": "object",
            "type": "update",
            "criteria": {"_id": "upserted_id"},
            "data": {"field": "value2"},
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    producer.producer.flush()
    collection = step.db_client.connection.database["object"]
    collection.insert_one({"_id": "upserted_id", "field": "value0"})
    result = collection.find_one({"_id": "upserted_id"})
    assert result is not None
    assert result["field"] == "value0"
    step.start()
    result = collection.find_one({"_id": "upserted_id"})
    assert result is not None
    assert result["field"] == "value2"


def test_upsert_into_database(setUp):
    step, producer = setUp
    command = json.dumps(
        {
            "collection": "object",
            "type": "update",
            "criteria": {"_id": "upserted_id"},
            "data": {"field": "new_value"},
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    producer.produce({"payload": command})
    producer.producer.flush(1)
    step.start()
    collection = step.db_client.connection.database["object"]
    result = collection.find_one({"_id": "upserted_id"})
    assert result is not None
    assert result["field"] == "new_value"


def test_upsert_only_into_database(setUp):
    step, producer = setUp
    command = json.dumps(
        {
            "collection": "object",
            "type": "update",
            "criteria": {"_id": "upserted_only_id"},
            "data": {"field": "some_value"},
            "options": {"upsert": True, "set_on_insert": True},
        }
    )
    producer.produce({"payload": command})
    command = json.dumps(
        {
            "collection": "object",
            "type": "update",
            "criteria": {"_id": "upserted_only_id"},
            "data": {"field": "other_value"},
            "options": {"upsert": True, "set_on_insert": True},
        }
    )
    producer.produce({"payload": command})
    producer.produce({"payload": command})
    producer.producer.flush(1)
    step.start()
    collection = step.db_client.connection.database["object"]
    result = collection.find_one({"_id": "upserted_only_id"})
    assert result is not None
    assert result["field"] == "some_value"


def test_insert_probabilities_into_database(setUp):
    step, producer = setUp
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
    producer.produce({"payload": command})
    producer.producer.flush(1)
    producer.produce({"payload": command})
    step.start()
    collection = step.db_client.connection.database["object"]
    result = collection.find_one({"_id": "insert_probabilities_id"})
    assert result is not None
    assert len(result["probabilities"]) == 1
    probabilities = result["probabilities"][0]
    assert probabilities["classifier_name"] == "classifier"
    assert probabilities["version"] == "1"
    assert probabilities["class_rank_1"] == "class2"
    assert probabilities["probability_rank_1"] == 0.9
    assert len(probabilities["values"]) == 2


def test_update_probabilities_into_database(setUp):
    step, producer = setUp
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
    producer.produce({"payload": command})
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
    producer.produce({"payload": command})
    producer.produce({"payload": command})
    producer.producer.flush(1)
    step.start()
    collection = step.db_client.connection.database["object"]
    result = collection.find_one({"_id": "update_probabilities_id"})
    assert result is not None
    assert len(result["probabilities"]) == 1
    probabilities = result["probabilities"][0]
    assert probabilities["classifier_name"] == "classifier"
    assert probabilities["version"] == "1"
    assert probabilities["class_rank_1"] == "class1"
    assert probabilities["probability_rank_1"] == 0.5


def test_update_features_into_database(setUp):
    step, producer = setUp
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
    producer.produce({"payload": command})
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
    producer.produce({"payload": command})
    producer.produce({"payload": command})
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
    producer.produce({"payload": command})
    producer.producer.flush(1)
    step.start()
    collection = step.db_client.connection.database["object"]
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


def test_insert_detections(setUp):
    step, producer = setUp
    command = json.dumps(
        {
            "collection": "detection",
            "type": "update",
            "criteria": {"oid": "ZTF12345", "candid": "12345"},
            "data": {
                "fid": "g",
                "mjd": 55500,
                # only some data is needed for the test
            },
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    command = json.dumps(
        {
            "collection": "detection",
            "type": "update",
            "criteria": {"oid": "ZTF45678", "candid": "12345"},
            "data": {
                "fid": "g",
                "mjd": 56500,
                # only some data is needed for the test
                # this is a different oid
            },
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    producer.producer.flush()
    step.start()
    collection = step.db_client.connection.database["detection"]
    result = collection.find({"candid": "12345", "oid": "ZTF12345"})
    result = list(result)
    assert len(result) == 1
    assert result[0]["oid"] == "ZTF12345"
    result = collection.find({"candid": "12345", "oid": "ZTF45678"})
    result = list(result)
    assert len(result) == 1
    assert result[0]["oid"] == "ZTF45678"


def test_insert_detections_with_conflict(setUp):
    step, producer = setUp
    command = json.dumps(
        {
            "collection": "detection",
            "type": "update",
            "criteria": {"oid": "ZTF12345", "candid": "12345"},
            "data": {
                "fid": "g",
                "mjd": 55500,
                # only some data is needed for the test
            },
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    command = json.dumps(
        {
            "collection": "detection",
            "type": "update",
            "criteria": {"oid": "ZTF45678", "candid": "12345"},
            "data": {
                "fid": "g",
                "mjd": 56500,
                # only some data is needed for the test
                # this is a different oid
            },
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    producer.producer.flush()
    collection = step.db_client.connection.database["detection"]
    collection.insert_one({"oid": "ZTF12345", "candid": "12345", "mjd": 55500})
    step.start()
    result = collection.find({"candid": "12345", "oid": "ZTF12345"})
    result = list(result)
    assert len(result) == 1
    assert result[0]["oid"] == "ZTF12345"
    result = collection.find({"candid": "12345", "oid": "ZTF45678"})
    result = list(result)
    assert len(result) == 1
    assert result[0]["oid"] == "ZTF45678"


def test_set_on_insert_detections(setUp):
    step, producer = setUp
    command = json.dumps(
        {
            "collection": "detection",
            "type": "update",
            "criteria": {"oid": "ZTF12345", "candid": "12345"},
            "data": {
                "fid": "g",
                "mjd": 55500,
                "has_stamp": True
                # only some data is needed for the test
            },
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    command = json.dumps(
        {
            "collection": "detection",
            "type": "update",
            "criteria": {"oid": "ZTF12345", "candid": "12345"},
            "data": {
                "fid": "g",
                "mjd": 56500,
                "has_stamp": False
                # only some data is needed for the test
                # this is a different oid
            },
            "options": {"upsert": True, "set_on_insert": True},
        }
    )
    producer.produce({"payload": command})
    producer.producer.flush()
    step.start()
    collection = step.db_client.connection.database["detection"]
    result = collection.find({"candid": "12345", "oid": "ZTF12345"})
    result = list(result)
    assert len(result) == 1
    assert result[0]["mjd"] == 55500


def test_insert_forced_photometry(setUp):
    step, producer = setUp
    command = json.dumps(
        {
            "collection": "forced_photometry",
            "type": "update",
            "criteria": {"oid": "ZTF12345", "candid": "ZTF1111"},
            "data": {
                "fid": "g",
                "mjd": 55500,
                "has_stamp": False,
                "pid": 1111
                # only some data is needed for the test
            },
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    producer.producer.flush()
    step.start()
    collection = step.db_client.connection.database["forced_photometry"]
    result = collection.find({"oid": "ZTF12345"})
    result = list(result)
    assert len(result) == 1
    assert "candid" not in result[0]


def test_update_non_detections_into_database(setUp):
    step, producer = setUp
    command = json.dumps(
        {
            "collection": "non_detection",
            "type": "update",
            "criteria": {
                "oid": "ZTF12345",
                "fid": "g",
                "mjd": 55500,
            },
            "data": {"diffmaglim": 21.11},
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    command = json.dumps(
        {
            "collection": "non_detection",
            "type": "update",
            "criteria": {
                "oid": "ZTF12345",
                "fid": "g",
                "mjd": 55500,
            },
            "data": {"diffmaglim": 33.33},
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    producer.produce({"payload": command})
    command = json.dumps(
        {
            "collection": "non_detection",
            "type": "update",
            "criteria": {
                "oid": "ZTF12345",
                "fid": "g",
                "mjd": 55500,
            },
            "data": {"diffmaglim": 11.11},
            "options": {"upsert": True},
        }
    )
    producer.produce({"payload": command})
    producer.producer.flush(1)
    step.start()
    collection = step.db_client.connection.database["non_detection"]
    result = collection.find_one({"oid": "ZTF12345"})
    assert result


def test_print_into_console(setUp):
    step, producer = setUp
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
    producer.produce({"payload": commands[0]})
    producer.produce({"payload": commands[1]})
    producer.producer.flush(1)
    step.start()
    os.environ["MOCK_DB_COLLECTION"] = ""
    collection = step.db_client.connection.database["object"]
    result = collection.find({"_id": "some printed value"})
    assert len(list(result)) == 0
