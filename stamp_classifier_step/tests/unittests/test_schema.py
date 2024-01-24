from apf.producers import KafkaProducer

from schema import SCHEMA


def test_schema():
    KafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": "test"},
            "TOPIC": "test",
            "SCHEMA": SCHEMA,
        }
    )
