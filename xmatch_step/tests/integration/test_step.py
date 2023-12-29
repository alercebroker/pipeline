import pathlib
import pytest

from xmatch_step import XmatchStep
from xmatch_step.core.xmatch_client import XmatchClient
from unittest import mock
from tests.data.messages import (
    generate_input_batch,
    get_fake_xmatch,
)
from apf.producers import KafkaProducer
from copy import deepcopy

PRODUCER_SCHEMA_PATH = pathlib.Path(
    pathlib.Path(__file__).parent.parent.parent.parent,
    "schemas/xmatch_step",
    "output.avsc",
)
SCRIBE_SCHEMA_PATH = pathlib.Path(
    pathlib.Path(__file__).parent.parent.parent.parent,
    "schemas/scribe_step",
    "scribe.avsc",
)

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "group_id",
        "auto.offset.reset": "beginning",
        "enable.partition.eof": True,
    },
    "TOPICS": ["correction"],
    "consume.messages": 20,
    "consume.timeout": 10,
}

PRODUCER_CONFIG = {
    "CLASS": "apf.producers.KafkaProducer",
    "TOPIC": "xmatch",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "apf.producers.KafkaProducer",
    "TOPIC": "w_object",
    "PARAMS": {"bootstrap.servers": "localhost:9092"},
    "SCHEMA_PATH": SCRIBE_SCHEMA_PATH,
}

XMATCH_CONFIG = {
    "CATALOG": {
        "name": "allwise",
        "columns": [
            "AllWISE",
            "RAJ2000",
            "DEJ2000",
            "W1mag",
            "W2mag",
            "W3mag",
            "W4mag",
            "e_W1mag",
            "e_W2mag",
            "e_W3mag",
            "e_W4mag",
            "Jmag",
            "e_Jmag",
            "Hmag",
            "e_Hmag",
            "Kmag",
            "e_Kmag",
        ],
    }
}


@pytest.fixture
def setUp() -> None:
    def default_handle_messages(batch):
        return batch

    def _setUp(handle_messages=default_handle_messages, extra_config={}):
        step_config = {
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
            "XMATCH_CONFIG": XMATCH_CONFIG,
            "RETRIES": 3,
            "RETRY_INTERVAL": 1,
        }
        step_config.update(extra_config)
        step = XmatchStep(config=step_config)
        batch = generate_input_batch(20)
        batch = handle_messages(batch)
        producer_config = {
            "CLASS": "apf.producers.KafkaProducer",
            "TOPIC": "correction",
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "SCHEMA_PATH": "../schemas/correction_step/output.avsc",
        }
        producer = KafkaProducer(config=producer_config)
        for item in batch:
            producer.produce(item)
        producer.producer.flush()
        return step, batch

    return _setUp


def test_step(kafka_service, setUp, kafka_consumer):
    step, batch = setUp()
    mock_xmatch = mock.Mock()
    mock_xmatch.return_value = get_fake_xmatch(batch)
    XmatchClient.execute = mock_xmatch
    step.start()
    consumer = kafka_consumer(["xmatch"])
    messages = list(consumer.consume())
    assert len(messages) == 20


def test_step_duplicate_objects(kafka_service, setUp, kafka_consumer):
    def repeat_oids(batch):
        return batch + batch

    consumer_config = deepcopy(CONSUMER_CONFIG)
    consumer_config["consume.messages"] = 40
    step, batch = setUp(repeat_oids, {"CONSUMER_CONFIG": consumer_config})
    mock_xmatch = mock.Mock()
    mock_xmatch.return_value = get_fake_xmatch(batch)
    XmatchClient.execute = mock_xmatch
    step.start()
    consumer = kafka_consumer(["xmatch"])
    messages = list(consumer.consume())
    assert len(messages) == 20
