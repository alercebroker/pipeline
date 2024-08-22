from typing import List
from apf.consumers import KafkaConsumer
from apf.producers import KafkaProducer
import tensorflow as tf
import pytest
import os

if tf.__version__.startswith("1"):
    pytest.skip("Incompatible TensorFlow version", allow_module_level=True)


from stamp_classifier_step.step import StampClassifierStep
from stamp_classifier_step.strategies.atlas import ATLASStrategy
from schema import SCHEMA, SCRIBE_SCHEMA


PRODUCER_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "test_schema.avsc")
SCRIBE_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "scribe.avsc")


def consume_messages(topic) -> List[dict]:
    config = {
        "PARAMS": {
            "bootstrap.servers": "localhost:9092",
            "group.id": "assert",
            "auto.offset.reset": "beginning",
            "max.poll.interval.ms": 36000,
            "session.timeout.ms": 36000,
            "enable.partition.eof": True,
        },
        "consume.timeout": 10,
        "consume.messages": 1,
        "TOPICS": [topic],
    }
    consumer = KafkaConsumer(config)
    messages = []
    # if len(consumer.consumer.assignment()) == 0:
    #    return messages

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
    assert '"type": "update_probabilities"' in messages[0]["payload"]
    assert f'"criteria": {{"_id": "{aid}"}}' in messages[0]["payload"]


@pytest.mark.usefixtures("kafka_service")
def test_atlas_step():
    consumer = KafkaConsumer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
                "group.id": "test-id",
                "auto.offset.reset": "beginning",
                "max.poll.interval.ms": 36000,
                "session.timeout.ms": 36000,
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
            "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
        }
    )

    scribe_producer = KafkaProducer(
        {
            "TOPIC": "object",
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "SCHEMA_PATH": SCRIBE_SCHEMA_PATH,
        }
    )
    strategy = ATLASStrategy()
    step = StampClassifierStep(
        consumer=consumer,
        producer=producer,
        scribe_producer=scribe_producer,
        strategy=strategy,
    )
    step.start()
    assert_messages_produced()
    assert_scribe_messages_produced()
