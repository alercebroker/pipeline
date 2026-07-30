import logging
from unittest import mock

import pytest
from prometheus_client import CollectorRegistry

from apf.core.step import GenericStep
from apf.metrics.prometheus import PrometheusMetrics
from apf.producers import GenericProducer
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


class RecordingProducer(GenericProducer):
    """Producer that writes its flush call to a shared list of calls."""

    def __init__(self, calls, name):
        super().__init__(config={})
        self.calls = calls
        self.name = name

    def produce(self, message=None, **kwargs):
        pass

    def flush(self):
        self.calls.append(f"flush:{self.name}")


class FailingProducer(RecordingProducer):
    """Producer with messages left undelivered when the flush gives up."""

    def flush(self):
        super().flush()
        raise BufferError("1 message(s) still undelivered")


class ScribeStep(MockStep):
    """Step that creates its own secondary producer, like the scribe steps."""

    def __init__(self, calls, scribe_producer=None, **kwargs):
        super().__init__(**kwargs)
        self.calls = calls
        self.scribe_producer = scribe_producer


def make_step(config, registry, calls, scribe_producer):
    step = ScribeStep(
        calls,
        scribe_producer=scribe_producer,
        config=config,
        metrics=PrometheusMetrics(registry=registry),
    )
    step.consumer.commit = lambda: calls.append("commit")
    return step


def test_scribe_producer_is_flushed_before_commit(
    config: dict, registry: CollectorRegistry
):
    calls = []
    step = make_step(config, registry, calls, RecordingProducer(calls, "scribe"))

    step._post_produce()

    assert calls == ["flush:scribe", "commit"]


def test_every_producer_is_flushed(config: dict, registry: CollectorRegistry):
    calls = []
    step = make_step(config, registry, calls, RecordingProducer(calls, "scribe"))
    step.producer = RecordingProducer(calls, "main")

    step._post_produce()

    assert calls == ["flush:main", "flush:scribe", "commit"]


def test_raising_flush_prevents_the_commit(
    config: dict, registry: CollectorRegistry, caplog
):
    caplog.set_level(logging.DEBUG)
    calls = []
    step = make_step(config, registry, calls, FailingProducer(calls, "scribe"))

    with pytest.raises(BufferError):
        step._post_produce()

    assert "commit" not in calls
    assert "will not be committed" in caplog.text
    assert registry.get_sample_value("exceptions_total", {"method": "flush"}) == 1


@mock.patch.object(GenericStep, "_write_success")
def test_flush_runs_when_the_producer_is_skipped(
    _, config: dict, registry: CollectorRegistry
):
    config["SKIP_PRODUCER"] = True
    calls = []
    step = make_step(config, registry, calls, RecordingProducer(calls, "scribe"))

    step.start()

    assert calls == ["flush:scribe", "commit"]


def test_unconfigured_scribe_producer_is_ignored(
    config: dict, registry: CollectorRegistry
):
    calls = []
    step = make_step(config, registry, calls, None)

    assert step._get_producers() == [step.producer]
    assert step.metrics_sender not in step._get_producers()

    step._post_produce()

    assert calls == ["commit"]


@mock.patch.object(GenericStep, "_write_success")
@mock.patch.object(GenericStep, "pre_execute")
def test_empty_batch_commits_without_flushing(
    pre_execute_mock, _, config: dict, registry: CollectorRegistry
):
    pre_execute_mock.return_value = []
    calls = []
    step = make_step(config, registry, calls, RecordingProducer(calls, "scribe"))

    step.start()

    assert calls == ["commit"]
