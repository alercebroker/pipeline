from fastavro.utils import generate_one
from prv_candidates_step.core.extractor import PreviousCandidatesExtractor
from tests.mocks.mock_alerts import extra_fields_generator
from tests.shared.sorting_hat_schema import SCHEMA


def test_compute_ztf():
    alert = generate_one(SCHEMA)
    alert["sid"] = "ZTF"
    alert["extra_fields"] = extra_fields_generator()
    result = PreviousCandidatesExtractor([alert]).extract_all()
    assert len(result[0]["detections"]) == 3
    assert len(result[0]["non_detections"]) == 2
    assert result[0]["detections"][1]["aid"] == alert["aid"]
    assert result[0]["detections"][2]["aid"] == alert["aid"]
    assert result[0]["detections"][0]["extra_fields"]["parent_candid"] is None
    assert result[0]["detections"][1]["extra_fields"]["parent_candid"] == alert["candid"]
    assert result[0]["detections"][2]["extra_fields"]["parent_candid"] == alert["candid"]
    assert result[0]["detections"][0]["has_stamp"]
    assert not result[0]["detections"][1]["has_stamp"]
    assert not result[0]["detections"][2]["has_stamp"]


def test_compute_atlas():
    alert = generate_one(SCHEMA)
    alert["sid"] = "ATLAS"
    alert["extra_fields"] = extra_fields_generator()
    result = PreviousCandidatesExtractor([alert]).extract_all()
    assert len(result[0]["detections"]) == 1
    assert len(result[0]["non_detections"]) == 0
    assert result[0]["detections"][0]["has_stamp"]
