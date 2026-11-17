import pytest
from apf.consumers import GenericConsumer

from hips_dummy_step.client import FetchResult
from hips_dummy_step.step import HipsDummyStep

from .conftest import URL


class FakeClient:
    params = {"bands": "r", "headers": "false"}

    def __init__(self):
        self.calls = []

    def fetch(self, positions):
        self.calls.append(list(positions))
        return FetchResult(
            n_positions=len(positions), n_failed=0, n_bytes=10, n_retries=0
        )


def make_consumer(batches):
    class ListConsumer(GenericConsumer):
        commits = 0

        def consume(self):
            yield from batches

        def commit(self):
            ListConsumer.commits += 1

    return ListConsumer


def ztf_alert(candid, ra, dec):
    return {"candid": candid, "candidate": {"candid": candid, "ra": ra, "dec": dec}}


def lsst_alert(source_id, ra, dec):
    return {"diaSourceId": source_id, "diaSource": {"ra": ra, "dec": dec}}


def make_step(survey, client, service_metrics, batches=(), max_positions=2):
    config = {
        "CONSUMER_CONFIG": {"PARAMS": {}},
        "SURVEY": survey,
        "IMAGE_SERVICE_CONFIG": {
            "URL": URL,
            "MAX_POSITIONS_PER_REQUEST": max_positions,
        },
    }
    return HipsDummyStep(
        config=config,
        client=client,
        service_metrics=service_metrics,
        consumer=make_consumer(list(batches)),
    )


def test_ztf_batch_is_split_into_requests(service_metrics, registry):
    client = FakeClient()
    step = make_step("ztf", client, service_metrics, max_positions=2)
    alerts = [
        ztf_alert(1, 10.0, 1.0),
        ztf_alert(2, 360.0, 2.0),
        ztf_alert(3, None, 3.0),
        ztf_alert(4, 40.0, 4.0),
    ]

    positions = step.pre_execute(alerts)
    assert step.execute(positions) == []

    assert client.calls == [[(10.0, 1.0), (0.0, 2.0)], [(40.0, 4.0)]]
    assert (
        registry.get_sample_value("image_service_positions_skipped_total") == 1
    )


def test_lsst_alerts_use_the_dia_source(service_metrics):
    client = FakeClient()
    step = make_step("LSST", client, service_metrics, max_positions=10)

    step.execute(step.pre_execute([lsst_alert(1, 5.0, -5.0)]))

    assert client.calls == [[(5.0, -5.0)]]


def test_start_requests_every_alert_and_commits(
    service_metrics, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # the framework writes a __SUCCESS__ file
    client = FakeClient()
    batches = [
        [ztf_alert(1, 1.0, 1.0), ztf_alert(2, 2.0, 2.0)],
        [ztf_alert(3, 3.0, 3.0)],
        [ztf_alert(4, float("nan"), 4.0)],
    ]
    step = make_step("ztf", client, service_metrics, batches=batches)

    step.start()

    assert client.calls == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]
    assert step.consumer.commits == 3


def test_unknown_survey_is_rejected(service_metrics):
    with pytest.raises(ValueError, match="SURVEY"):
        make_step("atlas", FakeClient(), service_metrics)
