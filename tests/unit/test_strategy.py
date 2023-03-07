from fastavro.utils import generate_one
from prv_candidates_step.core import (
    ZTFPrvCandidatesStrategy,
    ATLASPrvCandidatesStrategy,
)
from tests.mocks.mock_alerts import extra_fields_generator
from tests.shared.sorting_hat_schema import SCHEMA


def test_compute_ztf():
    strat = ZTFPrvCandidatesStrategy()
    alert = generate_one(SCHEMA)
    alert["extra_fields"] = extra_fields_generator()
    prv_detections, non_detections = strat.process_prv_candidates(alert)
    assert len(prv_detections) == 2
    assert len(non_detections) == 2


def test_compute_atlas():
    strat = ATLASPrvCandidatesStrategy()
    alert = generate_one(SCHEMA)
    alert["extra_fields"] = extra_fields_generator()
    prv_detections, non_detections = strat.process_prv_candidates(alert)
    assert len(prv_detections) == 0
    assert len(non_detections) == 0
