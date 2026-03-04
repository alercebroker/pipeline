import logging
from unittest import mock

import pytest

from apf.core.step import (
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


@mock.patch.object(GenericStep, "pre_execute")
def test_pre_execute_error(pre_execute_mock, step, caplog):
    caplog.set_level(logging.DEBUG)
    pre_execute_mock.side_effect = Exception("errorsito")
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at pre_execute" in caplog.text
    assert "The message(s) that caused the error: [{}]" in caplog.text


@mock.patch("conftest.MockStep.execute")
def test_execute_error(execute_mock, step, caplog):
    caplog.set_level(logging.DEBUG)
    execute_mock.side_effect = Exception("errorsito")
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at execute" in caplog.text
    assert "The message(s) that caused the error: [{}]" in caplog.text


@mock.patch.object(GenericStep, "post_execute")
def test_post_execute_error(post_execute_mock, step, caplog):
    caplog.set_level(logging.DEBUG)
    post_execute_mock.side_effect = Exception("errorsito")
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at post_execute" in caplog.text
    assert "The result that caused the error:" in caplog.text


@mock.patch.object(GenericStep, "pre_produce")
def test_pre_produce_error(pre_produce_mock, step, caplog):
    caplog.set_level(logging.DEBUG)
    pre_produce_mock.side_effect = Exception("errorsito")
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at pre_produce" in caplog.text
    assert "The result that caused the error:" in caplog.text


@mock.patch.object(GenericStep, "post_produce")
def test_post_produce_error(post_produce_mock, step, caplog):
    caplog.set_level(logging.DEBUG)
    post_produce_mock.side_effect = Exception("errorsito")
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at post_produce" in caplog.text


@mock.patch("conftest.MockStep.execute")
def test_nested_error(execute_mock, step, caplog):
    caplog.set_level(logging.DEBUG)

    def fun1(*args, **kwargs):
        def fun2():
            def fun3():
                raise Exception("errorsito")

            fun3()

        fun2()
        return {}

    execute_mock.side_effect = fun1
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at execute" in caplog.text
