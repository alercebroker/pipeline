from apf.core.step import GenericStep
import pytest
import requests


@pytest.fixture
def basic_config():
    return {
        "PROMETHEUS": True,
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


def test_init(basic_config):
    step = MockStep(config=basic_config)
    step._pre_consume()
    result = requests.get("http://localhost:8000")
    assert "consumed_messages summary" in result.text
    assert "processed_messages summary" in result.text
    assert "execution_time summary" in result.text
    assert "telescope_id gauge" in result.text
    del step


def test_consume(basic_config, mocker):
    mocker.patch.object(MockStep, "_write_success")
    step = MockStep(config=basic_config)
    step.start()
    result = requests.get("http://localhost:8000")
    assert "consumed_messages_count 1.0" in result.text
    assert "consumed_messages_sum 1.0" in result.text
    assert "processed_messages_count 1.0" in result.text
    assert "processed_messages_sum 1.0" in result.text
    assert "execution_time_count 1.0" in result.text
    del step
