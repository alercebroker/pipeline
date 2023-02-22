from datetime import datetime, timezone
import logging
from apf.core.step import (
    DefaultMetricsProducer,
    GenericStep,
    GenericProducer,
)
import pytest


@pytest.fixture
def basic_config():
    return {
        "CONSUMER_CONFIG": {
            "PARAMS": {},
            "CLASS": "apf.core.step.DefaultConsumer",
        },
        "PRODUCER_CONFIG": {
            "PARAMS": {},
            "CLASS": "apf.core.step.DefaultProducer",
        },
        "METRICS_CONFIG": {
            "CLASS": "apf.core.step.DefaultMetricsProducer",
            "PARAMS": {},
            "EXTRA_METRICS": ["oid", "candid"],
        },
    }


class MockStep(GenericStep):
    def execute(self, _):
        pass


def test_get_single_extra_metrics(basic_config):
    message = {"oid": "TEST", "candid": 1}
    gs = MockStep(config=basic_config)
    extra_metrics = gs.get_extra_metrics(message)
    assert type(extra_metrics) is dict
    assert type(extra_metrics["oid"]) is str
    assert type(extra_metrics["candid"]) is int
    assert type(extra_metrics["n_messages"]) is int


def test_get_batch_extra_metrics(basic_config):
    basic_config["METRICS_CONFIG"] = {
        "CLASS": "apf.core.step.DefaultMetricsProducer",
        "PARAMS": {},
        "EXTRA_METRICS": [
            "oid",
            "candid",
            {"key": "candid", "alias": "str_candid", "format": lambda x: str(x)},
        ],
    }
    message = [{"oid": "TEST", "candid": 1}, {"oid": "TEST2"}, {"candid": 3}]
    gs = MockStep(config=basic_config)
    extra_metrics = gs.get_extra_metrics(message)
    assert type(extra_metrics) is dict
    assert type(extra_metrics["oid"]) is list
    assert type(extra_metrics["candid"]) is list
    assert type(extra_metrics["n_messages"]) is int


def test_get_value(basic_config):
    basic_config["METRICS_CONFIG"] = {
        "CLASS": "apf.core.step.DefaultMetricsProducer",
        "PARAMS": {},
        "EXTRA_METRICS": ["oid", "candid"],
    }
    message = {"oid": "TEST", "candid": 1}
    gs = MockStep(config=basic_config)

    aliased_metric, value = gs.get_value(message, "oid")
    assert aliased_metric == "oid"
    assert value == "TEST"

    aliased_metric, value = gs.get_value(message, "candid")
    assert aliased_metric == "candid"
    assert value == 1

    aliased_metric, value = gs.get_value(message, {"key": "oid"})
    assert aliased_metric == "oid"
    assert value == "TEST"

    aliased_metric, value = gs.get_value(message, {"key": "oid", "alias": "new_oid"})
    assert aliased_metric == "new_oid"
    assert value == "TEST"

    aliased_metric, value = gs.get_value(
        message, {"key": "oid", "format": lambda x: x[0]}
    )
    assert aliased_metric == "oid"
    assert value == "T"

    with pytest.raises(KeyError):
        gs.get_value(message, {})

    with pytest.raises(ValueError):
        gs.get_value(message, {"key": "oid", "format": "test"})

    with pytest.raises(ValueError):
        gs.get_value(message, {"key": "oid", "alias": 1})


def test_without_consumer_config(basic_config):
    basic_config.update({"CONSUMER_CONFIG": {}})
    with pytest.raises(Exception):
        MockStep(config=basic_config)


def test_with_producer_config(basic_config):
    basic_config.update({"PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"}})
    gs = MockStep(config=basic_config)
    assert isinstance(gs.producer, GenericProducer)


def test_start(basic_config, mocker):
    write_mock = mocker.patch.object(MockStep, "_write_success")
    step = MockStep(config=basic_config, level=logging.DEBUG)
    step.start()
    write_mock.assert_called()


def test_pre_execute(basic_config, mocker):
    # mock the abstract method
    pre_execute = mocker.patch.object(MockStep, "pre_execute")
    step = MockStep(config=basic_config)
    message = {"msg": "message"}
    assert step.metrics.get("timestamp_received") == None
    step._pre_execute(message)
    pre_execute.assert_called_once_with(step.message)
    assert step.metrics.get("timestamp_received")
    assert step.message == [message]


def test_post_execute(basic_config, mocker):
    # mock the abstract method
    post_execute = mocker.patch.object(MockStep, "post_execute")
    send_metrics = mocker.patch.object(DefaultMetricsProducer, "send_metrics")
    result = {"msg": "message"}
    post_execute.return_value = result
    step = MockStep(config=basic_config)
    step.message = [result]
    step.metrics["timestamp_received"] = datetime.now(timezone.utc)
    assert step.metrics.get("timestamp_sent") == None
    assert step.metrics.get("execution_time") == None
    step._post_execute(result)
    post_execute.assert_called_once_with(result)
    assert step.metrics.get("timestamp_sent")
    send_metrics.assert_called()
