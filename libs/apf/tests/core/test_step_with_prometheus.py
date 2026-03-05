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
    for metric in registry.collect():
        print(metric)

    metrics = [
        "messages_consumed_total",
        "messages_processed_total",
        "batches_processed_total",
        "last_batch_processed_seconds",
        "last_execution_time_seconds",
    ]
    for metric in metrics:
        metric_value = registry.get_sample_value(metric)
        assert metric_value == 0, (
            f"Expected '{metric}' to start at '0'. Sampled '{metric_value}' instead."
        )


def test_consume(step: MockStep, registry: CollectorRegistry):
    step.start()

    assert registry.get_sample_value("messages_consumed_total") == 1
    assert registry.get_sample_value("messages_processed_total") == 1
    assert registry.get_sample_value("batches_processed_total") == 1
    assert registry.get_sample_value("last_batch_processed_seconds") > 0
    assert registry.get_sample_value("last_execution_time_seconds") > 0
    assert registry.get_sample_value("exceptions_total", {"method": "pre_execute"}) == 0
    assert registry.get_sample_value("exceptions_total", {"method": "execute"}) == 0
    assert (
        registry.get_sample_value("exceptions_total", {"method": "post_execute"}) == 0
    )
