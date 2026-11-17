import json

import pytest
import requests
import responses
from responses import matchers

from hips_dummy_step.client import ImageServiceClient

from .conftest import URL, npz_payload


@pytest.fixture
def client(service_metrics, sleeps):
    return ImageServiceClient(
        url=URL,
        bands=["g", "r"],
        headers=False,
        metrics=service_metrics,
        sleep=sleeps.append,
    )


def metric(registry, name, labels=None):
    return registry.get_sample_value(name, labels or {})


@responses.activate
def test_fetch_sends_positions_and_counts_failures(client, registry):
    responses.post(
        URL,
        body=npz_payload(3, failed=(1,)),
        match=[
            matchers.query_param_matcher({"bands": "g,r", "headers": "false"})
        ],
    )

    result = client.fetch([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    assert result.n_positions == 3
    assert result.n_failed == 1
    assert result.n_retries == 0
    assert json.loads(responses.calls[0].request.body) == [
        {"ra": 1.0, "dec": 2.0},
        {"ra": 3.0, "dec": 4.0},
        {"ra": 5.0, "dec": 6.0},
    ]
    assert metric(registry, "image_service_positions_requested_total") == 3
    assert metric(registry, "image_service_positions_failed_total") == 1
    assert metric(registry, "image_service_request_duration_seconds_count") == 1


@responses.activate
def test_busy_service_is_retried_after_the_time_it_asks(
    client, sleeps, registry
):
    responses.post(URL, status=429, headers={"Retry-After": "2"})
    responses.post(URL, status=429)  # no header: default wait
    responses.post(URL, body=npz_payload(1))

    result = client.fetch([(1.0, 2.0)])

    assert result.n_retries == 2
    assert sleeps == [2.0, 2.0]
    assert metric(registry, "image_service_retries_total", {"reason": "busy"}) == 2


@responses.activate
def test_server_errors_and_connection_errors_back_off(client, sleeps, registry):
    responses.post(URL, status=503)
    responses.post(URL, body=requests.ConnectionError("refused"))
    responses.post(URL, body=requests.Timeout("slow"))
    responses.post(URL, body=npz_payload(1))

    result = client.fetch([(1.0, 2.0)])

    assert result.n_retries == 3
    assert sleeps == [1.0, 2.0, 4.0]
    assert (
        metric(registry, "image_service_retries_total", {"reason": "server_error"})
        == 1
    )
    assert (
        metric(registry, "image_service_retries_total", {"reason": "connection"})
        == 2
    )


@responses.activate
def test_backoff_is_capped(service_metrics, sleeps):
    client = ImageServiceClient(
        url=URL, max_backoff=3.0, metrics=service_metrics, sleep=sleeps.append
    )
    for _ in range(4):
        responses.post(URL, status=500)
    responses.post(URL, body=npz_payload(1))

    client.fetch([(1.0, 2.0)])

    assert sleeps == [1.0, 2.0, 3.0, 3.0]


@responses.activate
def test_rejected_request_is_raised(client, sleeps):
    responses.post(URL, status=400, body="Invalid bands")

    with pytest.raises(RuntimeError, match="HTTP 400"):
        client.fetch([(1.0, 2.0)])
    assert sleeps == []


@responses.activate
def test_wrong_number_of_images_is_raised(client):
    responses.post(URL, body=npz_payload(1))

    with pytest.raises(RuntimeError, match="Asked for 2 positions"):
        client.fetch([(1.0, 2.0), (3.0, 4.0)])


@responses.activate
def test_optional_params_are_sent():
    client = ImageServiceClient(
        url=URL, bands=["r"], headers=True, reference_survey="some_survey"
    )
    responses.post(
        URL,
        body=npz_payload(1),
        match=[
            matchers.query_param_matcher(
                {"bands": "r", "headers": "true", "survey": "some_survey"}
            )
        ],
    )

    client.fetch([(1.0, 2.0)])


def test_url_comes_from_config_or_environment(monkeypatch):
    monkeypatch.delenv("IMAGE_SERVICE_URL", raising=False)
    assert ImageServiceClient.from_config({"URL": URL}).url == URL

    with pytest.raises(ValueError, match="IMAGE_SERVICE_URL"):
        ImageServiceClient.from_config({})

    monkeypatch.setenv("IMAGE_SERVICE_URL", URL)
    client = ImageServiceClient.from_config({"BANDS": ["g", "z"]})
    assert client.url == URL
    assert client.params == {"bands": "g,z", "headers": "false"}
