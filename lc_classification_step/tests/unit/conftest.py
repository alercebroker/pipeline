from json import loads
from unittest import mock

import pytest
from alerce_classifiers.base.dto import OutputDTO
from apf.consumers import KafkaConsumer
from apf.producers import KafkaProducer
from pandas import DataFrame

from lc_classification.core.step import LateClassifier
from tests.test_commons import (
    assert_command_is_correct,
    assert_elasticc_object_is_correct,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "ztf: mark a test as a ztf test.")
    config.addinivalue_line("markers", "elasticc: mark a test as a elasticc test.")


base_config = {
    "SCRIBE_PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test"},
    "PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test2"},
    "CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test3"},
    "MODEL_VERSION": "test",
    "SCRIBE_PARSER_CLASS": "lc_classification.core.parsers.scribe_parser.ScribeParser",
}


def ztf_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {"model": mock.MagicMock()},
            "CLASS": "lc_classification.predictors.ztf_random_forest.ztf_random_forest_predictor.ZtfRandomForestPredictor",
            "PARSER_CLASS": "lc_classification.predictors.ztf_random_forest.ztf_random_forest_parser.ZtfRandomForestParser",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.alerce_parser.AlerceParser",
    }


def toretto_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {"model_path": mock.MagicMock(), "model": mock.MagicMock()},
            "CLASS": "lc_classification.predictors.toretto.toretto_predictor.TorettoPredictor",
            "PARSER_CLASS": "lc_classification.predictors.toretto.toretto_parser.TorettoParser",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.elasticc_parser.ElasticcParser",
    }


def barney_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {"model_path": mock.MagicMock(), "model": mock.MagicMock()},
            "CLASS": "lc_classification.predictors.barney.barney_predictor.BarneyPredictor",
            "PARSER_CLASS": "lc_classification.predictors.barney.barney_parser.BarneyParser",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.elasticc_parser.ElasticcParser",
    }


def balto_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {
                "model_path": mock.MagicMock(),
                "model": mock.MagicMock(),
                "quantiles_path": mock.MagicMock(),
            },
            "CLASS": "lc_classification.predictors.balto.balto_predictor.BaltoPredictor",
            "PARSER_CLASS": "lc_classification.predictors.balto.balto_parser.BaltoParser",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.elasticc_parser.ElasticcParser",
    }


def messi_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {
                "model_path": mock.MagicMock(),
                "model": mock.MagicMock(),
                "header_quantiles_path": mock.MagicMock(),
                "feature_quantiles_path": mock.MagicMock(),
            },
            "CLASS": "lc_classification.predictors.messi.messi_predictor.MessiPredictor",
            "PARSER_CLASS": "lc_classification.predictors.messi.messi_parser.MessiParser",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.elasticc_parser.ElasticcParser",
    }


def step_factory(messages, config):
    step = LateClassifier(config=config)
    step.consumer = mock.MagicMock(KafkaConsumer)
    step.consumer.consume.return_value = messages
    step.producer = mock.MagicMock(KafkaProducer)
    step.scribe_producer = mock.create_autospec(KafkaProducer)
    return step


@pytest.fixture
def ztf_model_output():
    def output_factory(messages_ztf, model):
        aids = [
            message["aid"]
            for message in messages_ztf
            if message["features"] is not None
        ]
        model.predict_in_pipeline.return_value = {
            "hierarchical": {
                "top": DataFrame(
                    {
                        "aid": aids,
                        "CLASS": [1] * len(aids),
                        "CLASS2": [0] * len(aids),
                    }
                ),
                "children": {
                    "Transient": DataFrame({"aid": aids, "CLASS": [1] * len(aids)}),
                    "Stochastic": DataFrame({"aid": aids, "CLASS": [1] * len(aids)}),
                    "Periodic": DataFrame({"aid": aids, "CLASS": [1] * len(aids)}),
                },
            },
            "probabilities": DataFrame(
                {
                    "aid": aids,
                    "CLASS": [1] * len(aids),
                    "CLASS2": [0] * len(aids),
                }
            ),
        }

    return output_factory


@pytest.fixture
def elasticc_model_output():
    def factory(_, model):
        aids = ["aid1", "aid2"]
        df = DataFrame({"C1": [0.5, 0.9], "C2": [0.5, 0.1]}, index=aids)
        model.predict.return_value = OutputDTO(df)

    return factory


@pytest.fixture
def step_factory_ztf(ztf_model_output):
    def factory(messages_ztf):
        config = base_config.copy()
        config.update(ztf_config())
        ztf_model_output(messages_ztf, config["MODEL_CONFIG"]["PARAMS"]["model"])
        return step_factory(messages_ztf, config)

    return factory


@pytest.fixture
def step_factory_toretto(elasticc_model_output):
    def factory(messages):
        config = base_config.copy()
        config.update(toretto_config())
        elasticc_model_output(messages, config["MODEL_CONFIG"]["PARAMS"]["model"])
        step = step_factory(messages, config)
        step.step_parser.ClassMapper.set_mapping({"C1": 1, "C2": 2, "NotClassified": 3})
        return step

    return factory


@pytest.fixture
def step_factory_barney(elasticc_model_output):
    def factory(messages):
        config = base_config.copy()
        config.update(barney_config())
        elasticc_model_output(messages, config["MODEL_CONFIG"]["PARAMS"]["model"])
        step = step_factory(messages, config)
        step.step_parser.ClassMapper.set_mapping({"C1": 1, "C2": 2, "NotClassified": 3})
        return step

    return factory


@pytest.fixture
def step_factory_balto(elasticc_model_output):
    def factory(messages):
        config = base_config.copy()
        config.update(balto_config())
        elasticc_model_output(messages, config["MODEL_CONFIG"]["PARAMS"]["model"])
        step = step_factory(messages, config)
        step.step_parser.ClassMapper.set_mapping({"C1": 1, "C2": 2, "NotClassified": 3})
        return step

    return factory


@pytest.fixture
def step_factory_messi(elasticc_model_output):
    def factory(messages):
        config = base_config.copy()
        config.update(messi_config())
        elasticc_model_output(messages, config["MODEL_CONFIG"]["PARAMS"]["model"])
        step = step_factory(messages, config)
        step.step_parser.ClassMapper.set_mapping({"C1": 1, "C2": 2, "NotClassified": 3})
        return step

    return factory


@pytest.fixture
def test_elasticc_model():
    def test_model(factory, messages_elasticc):
        step = factory(messages_elasticc)
        step.start()
        predictor_calls = step.predictor.model.predict.mock_calls
        assert len(predictor_calls) > 0
        # Tests scribe produces correct commands
        scribe_calls = step.scribe_producer.mock_calls
        for call in scribe_calls:
            message = loads(call.args[0]["payload"])
            assert_command_is_correct(message)

        # Test producer produces correct data
        calls = step.producer.mock_calls
        assert len(calls) == len(messages_elasticc)
        for call in calls:
            obj = call.args[0]
            assert_elasticc_object_is_correct(obj)

    return test_model
