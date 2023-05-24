from unittest import mock
from lc_classification.core.elasticc_parser import ElasticcParser
from lc_classification.core.step import (
    LateClassifier,
)
from apf.producers import KafkaProducer
from apf.consumers import KafkaConsumer
from json import loads
from tests.mockdata.inputschema import INPUT_SCHEMA
from fastavro import utils


step_mock_config = {
    "SCRIBE_PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test"},
    "PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test2"},
    "CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test3"},
}

messages = utils.generate_many(INPUT_SCHEMA, 10)


def assert_object_is_correct(obj):
    assert "aid" in obj
    assert "candid" in obj
    assert "features" in obj
    assert "lc_classification" in obj


def assert_elasticc_object_is_correct(obj):
    assert "aid" in obj
    assert "candid" in obj
    assert "features" in obj
    assert "lc_classification" in obj
    assert "no_class" in obj["lc_classification"]["probabilities"]


def assert_command_is_correct(command):
    assert command["collection"] == "object"
    assert command["type"] == "update_probabilities"
    assert command["criteria"]["_id"] is not None
    assert "aid" not in command["data"]
    assert not command["options"]["set_on_insert"]


def test_step():
    step = LateClassifier(config=step_mock_config)
    step.consumer = mock.MagicMock(KafkaConsumer)
    step.consumer.consume.return_value = messages
    step.producer = mock.MagicMock(KafkaProducer)
    step.scribe_producer = mock.create_autospec(KafkaProducer)

    step.start()
    scribe_calls = step.scribe_producer.mock_calls

    # Tests scribe produces correct commands
    for call in scribe_calls:
        message = loads(call.args[0]["payload"])
        assert_command_is_correct(message)

    # Test producer produces correct data
    calls = step.producer.mock_calls
    for call in calls:
        obj = call.args[0]
        assert_object_is_correct(obj)


def test_step_elasticc():
    step = LateClassifier(config=step_mock_config, step_parser=ElasticcParser())
    step.consumer = mock.MagicMock(KafkaConsumer)
    step.consumer.consume.return_value = messages
    step.producer = mock.MagicMock(KafkaProducer)
    step.scribe_producer = mock.create_autospec(KafkaProducer)

    step.start()
    scribe_calls = step.scribe_producer.mock_calls

    # Tests scribe produces correct commands
    for call in scribe_calls:
        message = loads(call.args[0]["payload"])
        assert_command_is_correct(message)

    # Test producer produces correct data
    calls = step.producer.mock_calls
    for call in calls:
        obj = call.args[0]
        assert_elasticc_object_is_correct(obj)
