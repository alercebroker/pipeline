import os
from datetime import datetime, timezone
from unittest import mock

import pytest

from apf.core.step import (
    DefaultMetricsProducer,
    GenericProducer,
    GenericStep,
)
from tests.core.conftest import MockStep


@pytest.fixture
def config():
    return {
        "PROMETHEUS": False,
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


def test_get_single_extra_metrics(step):
    message = {"oid": "TEST", "candid": 1}
    extra_metrics = step.get_extra_metrics(message)
    assert type(extra_metrics) is dict
    assert type(extra_metrics["oid"]) is str
    assert type(extra_metrics["candid"]) is int
    assert type(extra_metrics["n_messages"]) is int


def test_get_batch_extra_metrics(config):
    config["METRICS_CONFIG"] = {
        "CLASS": "apf.core.step.DefaultMetricsProducer",
        "PARAMS": {},
        "EXTRA_METRICS": [
            "oid",
            "candid",
            {"key": "candid", "alias": "str_candid", "format": lambda x: str(x)},
        ],
    }
    message = [{"oid": "TEST", "candid": 1}, {"oid": "TEST2"}, {"candid": 3}]
    _step = MockStep(config=config)
    extra_metrics = _step.get_extra_metrics(message)
    assert type(extra_metrics) is dict
    assert type(extra_metrics["oid"]) is list
    assert type(extra_metrics["candid"]) is list
    assert type(extra_metrics["n_messages"]) is int
    del _step


def test_get_value(config):
    config["METRICS_CONFIG"] = {
        "CLASS": "apf.core.step.DefaultMetricsProducer",
        "PARAMS": {},
        "EXTRA_METRICS": ["oid", "candid"],
    }
    message = {"oid": "TEST", "candid": 1}
    step = MockStep(config=config)

    aliased_metric, value = step.get_value(message, "oid")
    assert aliased_metric == "oid"
    assert value == "TEST"

    aliased_metric, value = step.get_value(message, "candid")
    assert aliased_metric == "candid"
    assert value == 1

    aliased_metric, value = step.get_value(message, {"key": "oid"})
    assert aliased_metric == "oid"
    assert value == "TEST"

    aliased_metric, value = step.get_value(message, {"key": "oid", "alias": "new_oid"})
    assert aliased_metric == "new_oid"
    assert value == "TEST"

    aliased_metric, value = step.get_value(
        message, {"key": "oid", "format": lambda x: x[0]}
    )
    assert aliased_metric == "oid"
    assert value == "T"

    aliased_metric, value = step.get_value(message, {"key": "new_metric", "value": 1})
    assert aliased_metric == "new_metric"
    assert value == 1

    aliased_metric, value = step.get_value(
        message, {"key": "new_metric", "value": 1, "alias": "new_metric_alias"}
    )
    assert aliased_metric == "new_metric_alias"
    assert value == 1

    aliased_metric, value = step.get_value(
        message,
        {
            "key": "new_metric",
            "value": 1,
            "alias": "new_metric_alias",
            "format": lambda x: x + 1,
        },
    )
    assert aliased_metric == "new_metric_alias"
    assert value == 2

    with pytest.raises(KeyError):
        step.get_value(message, {})

    with pytest.raises(ValueError):
        step.get_value(message, {"key": "oid", "format": "test"})

    with pytest.raises(ValueError):
        step.get_value(message, {"key": "oid", "alias": 1})

    del step


def test_without_consumer_config(config):
    config.update({"CONSUMER_CONFIG": {}})
    with pytest.raises(Exception):
        MockStep(config=config)


def test_with_producer_config(config):
    config.update({"PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"}})
    step = MockStep(config=config)
    assert isinstance(step.producer, GenericProducer)
    del step


@mock.patch.object(GenericStep, "_write_success")
def test_start(write_mock, step: MockStep):
    step.start()
    write_mock.assert_called()


@mock.patch.object(GenericStep, "pre_execute")
def test_pre_execute(pre_execute_mock, step: MockStep):
    message = {"msg": "message"}
    assert step.kafka_metrics.get("timestamp_received") is None
    step._pre_execute([message])
    pre_execute_mock.assert_called_once_with(step.message)
    assert step.kafka_metrics.get("timestamp_received")
    assert step.message == [message]


@mock.patch.object(GenericStep, "post_execute")
@mock.patch.object(DefaultMetricsProducer, "send_metrics")
def test_post_execute(
    send_metrics_mock,
    post_execute_mock,
    step: MockStep,
):
    result = {"msg": "message"}
    post_execute_mock.return_value = result
    step.message = [result]
    step.kafka_metrics["timestamp_received"] = datetime.now(timezone.utc)
    assert step.kafka_metrics.get("timestamp_sent") is None
    assert step.kafka_metrics.get("execution_time") is None
    os.environ["METRICS_SURVEY"] = "test"
    step._post_execute(result)
    post_execute_mock.assert_called_once_with(result)
    assert step.kafka_metrics.get("timestamp_sent")
    assert step.kafka_metrics.get("source") == "MockStep"
    assert step.kafka_metrics.get("survey") == "test"
    send_metrics_mock.assert_called()
