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
    assert_ztf_object_is_correct,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "ztf: mark a test as a ztf test.")
    config.addinivalue_line(
        "markers", "elasticc: mark a test as a elasticc test."
    )


base_config = {
    "SCRIBE_PRODUCER_CONFIG": {
        "CLASS": "unittest.mock.MagicMock",
        "TOPIC": "test",
    },
    "PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test2"},
    "CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test3"},
    "MODEL_VERSION": "test",
    "SCRIBE_PARSER_CLASS": "lc_classification.core.parsers.scribe_parser.ScribeParser",
}


def ztf_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {},
            "CLASS": "lc_classifier.classifier.models.HierarchicalRandomForest",
            "NAME": "ZTF",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.alerce_parser.AlerceParser",
        "SCRIBE_PARSER_CLASS": "lc_classification.core.parsers.scribe_parser.ScribeParser",
    }


def toretto_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {
                "model_path": mock.MagicMock(),
            },
            "CLASS": "alerce_classifiers.rf_features_classifier.model.RandomForestFeaturesClassifier",
            "NAME": "toreto",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.elasticc_parser.ElasticcParser",
    }


def barney_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {
                "model_path": mock.MagicMock(),
            },
            "CLASS": "alerce_classifiers.rf_features_header_classifier.model.RandomForestFeaturesHeaderClassifier",
            "NAME": "barney",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.elasticc_parser.ElasticcParser",
    }


def balto_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {
                "model_path": mock.MagicMock(),
                "quantiles_path": mock.MagicMock(),
            },
            "CLASS": "alerce_classifiers.balto.model.BaltoClassifier",
            "MAPPER_CLASS": "alerce_classifiers.balto.mapper.BaltoMapper",
            "NAME": "balto",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.elasticc_parser.ElasticcParser",
    }


def anomaly_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {
                "model_path": mock.MagicMock(),
                "feature_quantiles_path": mock.MagicMock(),
            },
            "CLASS": "alerce_classifiers.anomaly.model.AnomalyDetector",
            "MAPPER_CLASS": "alerce_classifiers.anomaly.mapper.AnomalyMapper",
            "NAME": "anomaly",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.anomaly_parser.AnomalyParser",
    }


def messi_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {
                "model_path": mock.MagicMock(),
                "header_quantiles_path": mock.MagicMock(),
                "feature_quantiles_path": mock.MagicMock(),
            },
            "CLASS": "alerce_classifiers.messi.model.MessiClassifier",
            "MAPPER_CLASS": "alerce_classifiers.messi.mapper.MessiMapper",
            "NAME": "messi",
        },
        "STEP_PARSER_CLASS": "lc_classification.core.parsers.elasticc_parser.ElasticcParser",
    }


def mlp_config():
    return {
        "MODEL_CONFIG": {
            "PARAMS": {
                "model_path": mock.MagicMock(),
                "header_quantiles_path": mock.MagicMock(),
                "feature_quantiles_path": mock.MagicMock(),
            },
            "CLASS": "alerce_classifiers.tinkywinky.model.TinkyWinkyClassifier",
            "MAPPER_CLASS": "alerce_classifiers.mlp_elasticc.mapper.MLPMapper",
            "NAME": "mlp",
        },
        "STEP_PARSER_CLASS": (
            "lc_classification.core.parsers.elasticc_parser.ElasticcParser"
        ),
    }


def step_factory(messages, config, model):
    step = LateClassifier(config=config, model=model)
    step.consumer = mock.MagicMock(KafkaConsumer)
    step.consumer.consume.return_value = messages
    step.producer = mock.MagicMock(KafkaProducer)
    step.scribe_producer = mock.create_autospec(KafkaProducer)
    return step


@pytest.fixture
def ztf_model_output():
    def output_factory(messages_ztf, model):
        oids = [
            message["oid"]
            for message in messages_ztf
            if message.get("features") is not None
        ]
        model.predict_in_pipeline.return_value = {
            "hierarchical": {
                "top": DataFrame(
                    {
                        "oid": oids,
                        "CLASS": [1] * len(oids),
                        "CLASS2": [0] * len(oids),
                    }
                ),
                "children": {
                    "Transient": DataFrame(
                        {"oid": oids, "CLASS": [1] * len(oids)}
                    ),
                    "Stochastic": DataFrame(
                        {"oid": oids, "CLASS": [1] * len(oids)}
                    ),
                    "Periodic": DataFrame(
                        {"oid": oids, "CLASS": [1] * len(oids)}
                    ),
                },
            },
            "probabilities": DataFrame(
                {
                    "oid": oids,
                    "CLASS": [1] * len(oids),
                    "CLASS2": [0] * len(oids),
                }
            ),
        }

    return output_factory


@pytest.fixture
def ztf_anomaly_model_output():
    def output_factory(messages_ztf, model):
        oids = [message["oid"] for message in messages_ztf]
        df = DataFrame(
            {
                "C1": [1.0] * len(oids),
                "C2": [0.0] * len(oids),
                "NotClassified": [0.0] * len(oids),
                "oid": oids,
            },
        )
        df.set_index("oid", inplace=True)
        model.predict.return_value = OutputDTO(
            df, {"top": DataFrame(), "children": {}}
        )

    return output_factory


@pytest.fixture
def elasticc_model_output():
    def factory(_, model):
        oids = ["oid1", "oid2"]
        df = DataFrame({"C1": [0.5, 0.9], "C2": [0.5, 0.1]}, index=oids)
        model.predict.return_value = OutputDTO(
            df, {"top": DataFrame(), "children": {}}
        )

    return factory


@pytest.fixture
def step_factory_ztf(ztf_model_output):
    def factory(messages_ztf):
        config = base_config.copy()
        config.update(ztf_config())
        model_mock = mock.MagicMock()
        model_mock.can_predict.return_value = (True, "")
        ztf_model_output(messages_ztf, model_mock)
        return step_factory(messages_ztf, config, model=model_mock)

    return factory


@pytest.fixture
def step_factory_toretto(elasticc_model_output):
    def factory(messages):
        config = base_config.copy()
        config.update(toretto_config())
        model_mock = mock.MagicMock()
        model_mock.can_predict.return_value = (True, "")
        elasticc_model_output(messages, model_mock)
        step = step_factory(messages, config, model=model_mock)
        step.step_parser.ClassMapper.set_mapping(
            {"C1": 1, "C2": 2, "NotClassified": 3}
        )
        return step

    return factory


@pytest.fixture
def step_factory_barney(elasticc_model_output):
    def factory(messages):
        config = base_config.copy()
        config.update(barney_config())
        model_mock = mock.MagicMock()
        model_mock.can_predict.return_value = (True, "")
        elasticc_model_output(messages, model_mock)
        step = step_factory(messages, config, model=model_mock)
        step.step_parser.ClassMapper.set_mapping(
            {"C1": 1, "C2": 2, "NotClassified": 3}
        )
        return step

    return factory


@pytest.fixture
def step_factory_balto(elasticc_model_output):
    def factory(messages):
        config = base_config.copy()
        config.update(balto_config())
        model_mock = mock.MagicMock()
        model_mock.can_predict.return_value = (True, "")
        elasticc_model_output(messages, model_mock)
        step = step_factory(messages, config, model=model_mock)
        step.step_parser.ClassMapper.set_mapping(
            {"C1": 1, "C2": 2, "NotClassified": 3}
        )
        return step

    return factory


@pytest.fixture
def step_factory_messi(elasticc_model_output):
    def factory(messages):
        config = base_config.copy()
        config.update(messi_config())
        model_mock = mock.MagicMock()
        model_mock.can_predict.return_value = (True, "")
        elasticc_model_output(messages, model_mock)
        step = step_factory(messages, config, model=model_mock)
        step.step_parser.ClassMapper.set_mapping(
            {"C1": 1, "C2": 2, "NotClassified": 3}
        )
        return step

    return factory


@pytest.fixture
def step_factory_mlp(elasticc_model_output):
    def factory(messages):
        config = base_config.copy()
        config.update(mlp_config())
        model_mock = mock.MagicMock()
        model_mock.can_predict.return_value = (True, "")
        elasticc_model_output(messages, model_mock)
        step = step_factory(messages, config, model=model_mock)
        step.step_parser.ClassMapper.set_mapping(
            {"C1": 1, "C2": 2, "NotClassified": 3}
        )
        return step

    return factory


@pytest.fixture
def step_factory_anomaly(ztf_anomaly_model_output):
    def factory(messages):
        config = base_config.copy()
        config.update(anomaly_config())
        model_mock = mock.MagicMock()
        model_mock.can_predict.return_value = (True, "")
        ztf_anomaly_model_output(messages, model_mock)
        step = step_factory(messages, config, model=model_mock)
        return step

    return factory


@pytest.fixture
def test_elasticc_model():
    def test_model(factory, messages_elasticc):
        step = factory(messages_elasticc)
        step.start()
        predictor_calls = step.model.predict.mock_calls
        assert len(predictor_calls) > 0
        # Tests scribe produces correct commands
        scribe_calls = step.scribe_producer.mock_calls
        for call in scribe_calls:
            message = loads(call.args[0]["payload"])
            assert_command_is_correct(message)

        # Test producer produces correct data
        calls = step.producer.mock_calls
        assert len(calls) == len(messages_elasticc) + 1  # beause call __del__
        for call in calls:
            if len(call.args) > 1:  # because of __del__
                obj = call.args[0]
                assert_elasticc_object_is_correct(obj)

    return test_model


@pytest.fixture
def test_anomaly_model():
    def test_model(factory, messages_ztf):
        step = factory(messages_ztf)
        step.start()
        predictor_calls = step.model.predict.mock_calls
        assert len(predictor_calls) > 0
        scribe_calls = step.scribe_producer.mock_calls
        for call in scribe_calls:
            message = loads(call.args[0]["payload"])
            assert_command_is_correct(message)

        # Test producer produces correct data
        calls = step.producer.mock_calls
        assert len(calls) == len(messages_ztf) + 1  # because call __del__
        for call in calls:
            if len(call.args) > 1:  # because of __del__
                obj = call.args[0]
                assert_ztf_object_is_correct(obj)

    return test_model
