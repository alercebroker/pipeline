from unittest import mock
from lc_classification.core.step import (
    LateClassifier,
)
from apf.producers import KafkaProducer
from apf.consumers import KafkaConsumer
from json import loads
from tests.mockdata.input_ztf import INPUT_SCHEMA as INPUT_ZTF
from tests.mockdata.input_elasticc import INPUT_SCHEMA as INPUT_ELASTICC
from fastavro import utils
import pytest
import os


step_mock_config = {
    "SCRIBE_PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test"},
    "PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test2"},
    "CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test3"},
    "MODEL_VERSION": "test",
}

messages_ztf = utils.generate_many(INPUT_ZTF, 10)
messages_elasticc = utils.generate_many(INPUT_ELASTICC, 10)


def assert_object_is_correct(obj):
    assert "aid" in obj
    assert "candid" in obj
    assert "features" in obj
    assert "lc_classification" in obj


def assert_elasticc_object_is_correct(obj):
    assert "classifications" in obj
    assert isinstance(obj["classifications"], list)
    assert len(obj["classifications"]) > 0


def assert_command_is_correct(command):
    assert command["collection"] == "object"
    assert command["type"] == "update_probabilities"
    assert command["criteria"]["_id"] is not None
    assert "aid" not in command["data"]
    assert not command["options"]["set_on_insert"]


@pytest.mark.skipif(os.getenv("MODEL") != "ztf", reason="ztf only")
def test_step():
    step_mock_config[
        "PREDICTOR_CLASS"
    ] = "lc_classification.predictors.ztf_random_forest.ztf_random_forest_predictor.ZtfRandomForestPredictor"
    step_mock_config[
        "PREDICTOR_PARSER_CLASS"
    ] = "lc_classification.predictors.ztf_random_forest.ztf_random_forest_parser.ZtfRandomForestParser"
    step_mock_config[
        "SCRIBE_PARSER_CLASS"
    ] = "lc_classification.core.parsers.scribe_parser.ScribeParser"
    step_mock_config[
        "STEP_PARSER_CLASS"
    ] = "lc_classification.core.parsers.alerce_parser.AlerceParser"
    step = LateClassifier(config=step_mock_config)
    step.consumer = mock.MagicMock(KafkaConsumer)
    step.consumer.consume.return_value = messages_ztf
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


@pytest.mark.skipif(os.getenv("MODEL") != "elasticc", reason="elasticc only")
def test_step_elasticc():
    model_path = "https://assets.alerce.online/pipeline/elasticc/random_forest/2.0.1/"
    step_mock_config[
        "PREDICTOR_CLASS"
    ] = "lc_classification.predictors.elasticc_random_forest.elasticc_random_forest_predictor.ElasticcRandomForestPredictor"
    step_mock_config["PREDICTOR_PARAMS"] = {"model_path": model_path}
    step_mock_config[
        "PREDICTOR_PARSER_CLASS"
    ] = "lc_classification.predictors.elasticc_random_forest.elasticc_random_forest_parser.ElasticcRandomForestParser"
    step_mock_config[
        "SCRIBE_PARSER_CLASS"
    ] = "lc_classification.core.parsers.scribe_parser.ScribeParser"
    step_mock_config[
        "STEP_PARSER_CLASS"
    ] = "lc_classification.core.parsers.elasticc_parser.ElasticcParser"
    step = LateClassifier(config=step_mock_config)
    step.consumer = mock.MagicMock(KafkaConsumer)
    step.consumer.consume.return_value = messages_elasticc
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
