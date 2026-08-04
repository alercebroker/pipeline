from apf.core.step import (
    GenericStep,
)
import pytest
import logging


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


def test_pre_execute_error(mocker, basic_config, caplog):
    caplog.set_level(logging.DEBUG)
    mock = mocker.patch.object(MockStep, "pre_execute")
    mock.side_effect = Exception("errorsito")
    step = MockStep(config=basic_config)
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at pre_execute" in caplog.text
    assert "The message(s) that caused the error: [{}]" in caplog.text


def test_execute_error(mocker, basic_config, caplog):
    caplog.set_level(logging.DEBUG)
    mock = mocker.patch.object(MockStep, "execute")
    mock.side_effect = Exception("errorsito")
    step = MockStep(config=basic_config)
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at execute" in caplog.text
    assert "The message(s) that caused the error: {}" in caplog.text


def test_post_execute_error(mocker, basic_config, caplog):
    caplog.set_level(logging.DEBUG)
    mock = mocker.patch.object(MockStep, "post_execute")
    mock.side_effect = Exception("errorsito")
    step = MockStep(config=basic_config)
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at post_execute" in caplog.text
    assert "The result that caused the error:" in caplog.text


def test_pre_produce_error(mocker, basic_config, caplog):
    caplog.set_level(logging.DEBUG)
    mock = mocker.patch.object(MockStep, "pre_produce")
    mock.side_effect = Exception("errorsito")
    step = MockStep(config=basic_config)
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at pre_produce" in caplog.text
    assert "The result that caused the error:" in caplog.text


def test_post_produce_error(mocker, basic_config, caplog):
    caplog.set_level(logging.DEBUG)
    mock = mocker.patch.object(MockStep, "post_produce")
    mock.side_effect = Exception("errorsito")
    step = MockStep(config=basic_config)
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at post_produce" in caplog.text


def test_nested_error(mocker, basic_config, caplog):
    caplog.set_level(logging.DEBUG)

    def fun1(*args, **kwargs):
        def fun2():
            def fun3():
                raise Exception("errorsito")

            fun3()

        fun2()
        return {}

    mock = mocker.patch.object(MockStep, "execute")
    mock.side_effect = fun1
    step = MockStep(config=basic_config)
    with pytest.raises(Exception) as error:
        step.start()
    assert "errorsito" in error.value.args[0]
    assert "Error at execute" in caplog.text
