import logging

import pytest

from apf.core.step import GenericStep
from apf.producers import GenericProducer


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


class ScribeStep(GenericStep):
    """Step that creates its own secondary producer, like the scribe steps."""

    def __init__(self, calls, scribe_producer=None, **kwargs):
        super().__init__(**kwargs)
        self.calls = calls
        self.scribe_producer = scribe_producer

    def execute(self, _):
        return {}


def make_step(config, calls, scribe_producer):
    step = ScribeStep(calls, scribe_producer=scribe_producer, config=config)
    step.consumer.commit = lambda: calls.append("commit")
    return step


def test_scribe_producer_is_flushed_before_commit(config: dict):
    calls = []
    step = make_step(config, calls, RecordingProducer(calls, "scribe"))

    step._post_produce()

    assert calls == ["flush:scribe", "commit"]


def test_every_producer_is_flushed(config: dict):
    calls = []
    step = make_step(config, calls, RecordingProducer(calls, "scribe"))
    step.producer = RecordingProducer(calls, "main")

    step._post_produce()

    assert calls == ["flush:main", "flush:scribe", "commit"]


def test_raising_flush_prevents_the_commit(config: dict, caplog):
    caplog.set_level(logging.DEBUG)
    calls = []
    step = make_step(config, calls, FailingProducer(calls, "scribe"))

    with pytest.raises(BufferError):
        step._post_produce()

    assert "commit" not in calls
    assert "will not be committed" in caplog.text


def test_unconfigured_scribe_producer_is_ignored(config: dict):
    calls = []
    step = make_step(config, calls, None)

    assert step._get_producers() == [step.producer]
    assert step.metrics_sender not in step._get_producers()

    step._post_produce()

    assert calls == ["commit"]


def test_commit_disabled_still_flushes(config: dict):
    config["COMMIT"] = False
    calls = []
    step = make_step(config, calls, RecordingProducer(calls, "scribe"))

    step._post_produce()

    assert calls == ["flush:scribe"]
