from prometheus_client import start_http_server
from apf.metrics.prometheus import PrometheusMetrics
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
        return {}


@pytest.fixture(scope="session")
def prometheus_server():
    start_http_server(8000)


prometheus_metrics = PrometheusMetrics()


@pytest.fixture
def step(basic_config, mocker, prometheus_server):
    mocker.patch.object(MockStep, "_write_success")
    step = MockStep(config=basic_config, prometheus_metrics=prometheus_metrics)
    yield step


def test_init(step):
    step._pre_consume()
    result = requests.get("http://localhost:8000")
    assert "consumed_messages summary" in result.text
    assert "processed_messages summary" in result.text
    assert "execution_time summary" in result.text
    # assert "telescope_id gauge" in result.text


def test_consume(step):
    step.start()
    result = requests.get("http://localhost:8000")
    assert "consumed_messages_count 1.0" in result.text
    assert "consumed_messages_sum 1.0" in result.text
    assert "processed_messages_count 1.0" in result.text
    assert "processed_messages_sum 1.0" in result.text
    assert "execution_time_count 1.0" in result.text
