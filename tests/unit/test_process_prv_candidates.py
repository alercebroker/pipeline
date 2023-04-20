from unittest import mock
from fastavro.utils import generate_many
from random import random
from tests.shared.sorting_hat_schema import SCHEMA
from unittest.mock import call

from prv_candidates_step.core.candidates.process_prv_candidates import (
    process_prv_candidates,
)


@mock.patch(
    "prv_candidates_step.core.candidates.process_prv_candidates.ZTFPrvCandidatesStrategy.process_prv_candidates"
)
@mock.patch(
    "prv_candidates_step.core.candidates.process_prv_candidates.ATLASPrvCandidatesStrategy.process_prv_candidates"
)
def test_process_prv_candidates(atlas_strat, ztf_strat):
    def flip_coin():
        if random() > 0.5:
            return "ATLAS"

        return "ZTF"

    alerts = list(generate_many(SCHEMA, 10))
    ztf_strat.return_value = ("prv_detections", "non_detections")
    atlas_strat.return_value = ("prv_detections", "non_detections")
    # generate valid tids
    for alert in alerts:
        alert["sid"] = flip_coin()

    result = process_prv_candidates(alerts)
    assert result == (["prv_detections"] * 10, ["non_detections"] * 10)
