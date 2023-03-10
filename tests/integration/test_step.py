from typing import List

from apf.consumers import KafkaConsumer
from apf.producers import KafkaProducer
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.mongo.models import Object

import tensorflow as tf
import pytest

if tf.__version__.startswith('1'):
    pytest.skip("Incompatible TensorFlow version", allow_module_level=True)


from stamp_classifier_step.step import StampClassifierStep
from stamp_classifier_step.strategies.atlas import ATLASStrategy
from schema import SCHEMA, SCRIBE_SCHEMA

from .conftest import generate_messages


DB_CONFIG = {
    "HOST": "localhost",
    "USERNAME": "root",
    "PASSWORD": "root",
    "PORT": 27017,
    "DATABASE": "test_db",
}


def consume_messages(topic) -> List[dict]:
    config = {
        "PARAMS": {
            "bootstrap.servers": "localhost:9092",
            "group.id": "assert",
            "auto.offset.reset": "beginning",
            "max.poll.interval.ms": 3600000,
            "enable.partition.eof": True,
        },
        "consume.timeout": 10,
        "consume.messages": 1,
        "TOPICS": [topic],
    }
    consumer = KafkaConsumer(config)
    messages = []
    # if len(consumer.consumer.assignment()) == 0:
    #     return messages

    for message in consumer.consume():
        messages.append(message)
    return messages


def assert_messages_produced(aid="aid"):
    messages = consume_messages("atlas_stamp_classifier")
    assert messages[0]["aid"] == aid
    assert messages[0]["classifications"][0]["class_name"] == "agn"


def assert_scribe_messages_produced(aid="aid"):
    messages = consume_messages("object")
    assert type(messages[0]["payload"]) == str
    assert '"collection": "object"' in messages[0]["payload"]
    assert '"type": "insert_probabilities"' in messages[0]["payload"]
    assert f'"criteria": {{"_id": "{aid}"}}' in messages[0]["payload"]


@pytest.mark.usefixtures("kafka_service")
@pytest.mark.usefixtures("mongo_service")
def test_atlas_step():
    consumer = KafkaConsumer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
                "group.id": "test-id",
                "auto.offset.reset": "beginning",
                "max.poll.interval.ms": 3600000,
                "enable.partition.eof": True,
            },
            "consume.timeout": 10,
            "consume.messages": 1,
            "TOPICS": ["sorting_hat"],
        }
    )
    producer = KafkaProducer(
        {
            "TOPIC": "atlas_stamp_classifier",
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "SCHEMA": SCHEMA,
        }
    )

    scribe_producer = KafkaProducer(
        {
            "TOPIC": "object",
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "SCHEMA": SCRIBE_SCHEMA,
        }
    )
    db_connection = MongoConnection(DB_CONFIG)
    db_connection.connect()
    strategy = ATLASStrategy()
    step = StampClassifierStep(
        consumer=consumer,
        producer=producer,
        scribe_producer=scribe_producer,
        db_connection=db_connection,
        strategy=strategy,
    )
    step.start()
    assert_messages_produced()
    assert_scribe_messages_produced()


@pytest.mark.usefixtures("kafka_service")
@pytest.mark.usefixtures("mongo_service")
def test_atlas_step_skips_objects_in_database():
    consumer = KafkaConsumer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
                "group.id": "test-id",
                "auto.offset.reset": "earliest",
                "max.poll.interval.ms": 3600000,
                "enable.partition.eof": True,
            },
            "consume.timeout": 10,
            "consume.messages": 2,
            "TOPICS": ["sorting_hat"],
        }
    )
    producer = KafkaProducer(
        {
            "TOPIC": "atlas_stamp_classifier",
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "SCHEMA": SCHEMA,
        }
    )

    scribe_producer = KafkaProducer(
        {
            "TOPIC": "object",
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "SCHEMA": SCRIBE_SCHEMA,
        }
    )
    db_connection = MongoConnection(DB_CONFIG)
    db_connection.connect()

    obj = {
        "oid": "oid",
        "tid": "tid",
        "corrected": False,
        "stellar": False,
        "firstmjd": 1,
        "lastmjd": 1,
        "ndet": 1,
        "meanra": 1,
        "sigmara": 1,
        "meandec": 1,
        "sigmadec": 1,
        "probabilities": [{"classifier_name": "atlas_stamp_classifier"}],
    }

    db_connection.query(Object).get_or_create({"_id": "aid1"}, **obj)

    generate_messages("aid1")

    strategy = ATLASStrategy()
    step = StampClassifierStep(
        consumer=consumer,
        producer=producer,
        scribe_producer=scribe_producer,
        db_connection=db_connection,
        strategy=strategy,
    )
    step.start()
    assert_messages_produced()
    assert_scribe_messages_produced()
