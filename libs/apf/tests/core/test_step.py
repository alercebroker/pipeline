from datetime import datetime, timezone
import os
from apf.core.step import (
    DefaultMetricsProducer,
    GenericStep,
    GenericProducer,
)
import pytest


class MockStep(GenericStep):
    def execute(self, _):
        return {}


@pytest.fixture
def basic_config():
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


@pytest.fixture
def step(basic_config, mocker):
    mocker.patch.object(MockStep, "_write_success")
    step = MockStep(config=basic_config)
    yield step
    del step


def test_get_single_extra_metrics(step):
    message = {"oid": "TEST", "candid": 1}
    extra_metrics = step.get_extra_metrics(message)
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
    _step = MockStep(config=basic_config)
    extra_metrics = _step.get_extra_metrics(message)
    assert type(extra_metrics) is dict
    assert type(extra_metrics["oid"]) is list
    assert type(extra_metrics["candid"]) is list
    assert type(extra_metrics["n_messages"]) is int
    del _step


def test_get_value(basic_config):
    basic_config["METRICS_CONFIG"] = {
        "CLASS": "apf.core.step.DefaultMetricsProducer",
        "PARAMS": {},
        "EXTRA_METRICS": ["oid", "candid"],
    }
    message = {"oid": "TEST", "candid": 1}
    step = MockStep(config=basic_config)

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


def test_without_consumer_config(basic_config):
    basic_config.update({"CONSUMER_CONFIG": {}})
    with pytest.raises(Exception):
        MockStep(config=basic_config)


def test_with_producer_config(basic_config):
    basic_config.update({"PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"}})
    step = MockStep(config=basic_config)
    assert isinstance(step.producer, GenericProducer)
    del step


def test_start(mocker, step):
    write_mock = mocker.patch.object(MockStep, "_write_success")
    step.start()
    write_mock.assert_called()


def test_pre_execute(mocker, step):
    # mock the abstract method
    pre_execute = mocker.patch.object(MockStep, "pre_execute")
    message = {"msg": "message"}
    assert step.metrics.get("timestamp_received") == None
    step._pre_execute(message)
    pre_execute.assert_called_once_with(step.message)
    assert step.metrics.get("timestamp_received")
    assert step.message == [message]


def test_post_execute(step, mocker):
    # mock the abstract method
    post_execute = mocker.patch.object(MockStep, "post_execute")
    send_metrics = mocker.patch.object(DefaultMetricsProducer, "send_metrics")
    result = {"msg": "message"}
    post_execute.return_value = result
    step.message = [result]
    step.metrics["timestamp_received"] = datetime.now(timezone.utc)
    assert step.metrics.get("timestamp_sent") == None
    assert step.metrics.get("execution_time") == None
    os.environ["METRICS_SURVEY"] = "test"
    step._post_execute(result)
    post_execute.assert_called_once_with(result)
    assert step.metrics.get("timestamp_sent")
    assert step.metrics.get("source") == "MockStep"
    assert step.metrics.get("survey") == "test"
    send_metrics.assert_called()


def test_start_with_execute_returning_iterable(basic_config, mocker):
    class StepWithIterableExecute(GenericStep):
        def execute(self, _):
            return [{}]

    write_mock = mocker.patch.object(StepWithIterableExecute, "_write_success")
    step = StepWithIterableExecute(config=basic_config)
    step.start()
    write_mock.assert_called()
