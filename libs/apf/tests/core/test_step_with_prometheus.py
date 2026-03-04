import pytest
from prometheus_client import CollectorRegistry

from tests.core.conftest import MockStep


@pytest.fixture
def config():
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


def test_init(step: MockStep, registry: CollectorRegistry):
    assert registry.get_sample_value("consumed_messages_count") == 0
    assert registry.get_sample_value("processed_messages_count") == 0
    assert registry.get_sample_value("execution_time_count") == 0


def test_consume(step: MockStep, registry: CollectorRegistry):
    step.start()

    assert registry.get_sample_value("consumed_messages_count") == 1
    assert registry.get_sample_value("consumed_messages_sum") == 1
    assert registry.get_sample_value("processed_messages_count") == 1
    assert registry.get_sample_value("processed_messages_sum") == 1
    assert registry.get_sample_value("execution_time_count") == 1
