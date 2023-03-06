from pytest_mock import MockerFixture
from fastavro.utils import generate_many
from tests.shared.sorting_hat_schema import SCHEMA
from unittest.mock import call

from prv_candidates_step.core.candidates.process_prv_candidates import (
    process_prv_candidates,
)


def test_process_prv_candidates(mocker: MockerFixture):
    processor = mocker.patch("prv_candidates_step.core.processor.processor.Processor")
    processor.compute.return_value = ("prv_detections", "non_detections")
    alerts = list(generate_many(SCHEMA, 10))
    result = process_prv_candidates(processor, alerts)
    processor.compute.assert_has_calls([call(alert) for alert in alerts])
    assert result == (["prv_detections"] * 10, ["non_detections"] * 10)
