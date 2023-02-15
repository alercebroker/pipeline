from apf.consumers import KafkaConsumer
from apf.producers import KafkaProducer
from atlas_stamp_classifier_step.step import AtlasStampClassifierStep
from atlas_stamp_classifier.inference import AtlasStampClassifier
from schema import SCHEMA, SCRIBE_SCHEMA
from typing import List


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
    for message in consumer.consume():
        messages.append(message)
    return messages


def assert_messages_produced():
    messages = consume_messages("atlas_stamp_classifier")
    assert messages[0]["aid"] == "aid"
    assert messages[0]["classifications"][0]["class_name"] == "agn"


def assert_scribe_messages_produced():
    messages = consume_messages("object")
    assert type(messages[0]["payload"]) == str
    assert '"collection": "object"' in messages[0]["payload"]
    assert '"type": "update-probabilities"' in messages[0]["payload"]
    assert '"criteria": {"aid": "aid"}' in messages[0]["payload"]


def test_step(kafka_service):
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
    model = AtlasStampClassifier()
    step = AtlasStampClassifierStep(
        consumer=consumer,
        producer=producer,
        scribe_producer=scribe_producer,
        config={},
        model=model,
    )
    step.start()
    assert_messages_produced()
    assert_scribe_messages_produced()
