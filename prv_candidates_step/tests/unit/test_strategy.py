import pathlib
from fastavro.utils import generate_one
from prv_candidates_step.core.extractor import PreviousCandidatesExtractor
from tests.mocks.mock_alerts import (
    ztf_extra_fields_generator,
    lsst_extra_fields_generator,
)
from fastavro.schema import load_schema

SCHEMA = load_schema(
    str(
        pathlib.Path(
            pathlib.Path(__file__).parent.parent.parent.parent,
            "schemas/sorting_hat_step",
            "output.avsc",
        )
    )
)


def test_compute_ztf():
    alert = generate_one(SCHEMA)
    alert["sid"] = "ZTF"
    alert["extra_fields"] = ztf_extra_fields_generator()
    result = PreviousCandidatesExtractor([alert]).extract_all()
    assert len(result[0]["detections"]) == 4
    assert len(result[0]["non_detections"]) == 2
    assert result[0]["detections"][1]["oid"] == alert["oid"]
    assert result[0]["detections"][2]["oid"] == alert["oid"]
    assert result[0]["detections"][0]["parent_candid"] is None
    assert result[0]["detections"][1]["parent_candid"] == alert["candid"]
    assert result[0]["detections"][2]["parent_candid"] == alert["candid"]
    assert result[0]["detections"][0]["has_stamp"]
    assert not result[0]["detections"][1]["has_stamp"]
    assert not result[0]["detections"][2]["has_stamp"]
    assert not result[0]["detections"][0]["forced"]
    assert not result[0]["detections"][1]["forced"]
    assert not result[0]["detections"][2]["forced"]


def test_compute_atlas():
    alert = generate_one(SCHEMA)
    alert["sid"] = "ATLAS"
    alert["extra_fields"] = {}
    result = PreviousCandidatesExtractor([alert]).extract_all()
    assert len(result[0]["detections"]) == 1
    assert len(result[0]["non_detections"]) == 0
    assert result[0]["detections"][0]["parent_candid"] is None
    assert result[0]["detections"][0]["has_stamp"]
    assert not result[0]["detections"][0]["forced"]


def test_compute_lsst():
    alert = generate_one(SCHEMA)
    alert["sid"] = "LSST"
    alert["extra_fields"] = lsst_extra_fields_generator()
    result = PreviousCandidatesExtractor([alert]).extract_all()
    assert len(result[0]["detections"]) == 3
    assert len(result[0]["non_detections"]) == 0
    assert result[0]["detections"][0]["parent_candid"] is None
    assert result[0]["detections"][1]["parent_candid"] == alert["candid"]
    assert result[0]["detections"][2]["parent_candid"] == alert["candid"]
    assert result[0]["detections"][0]["has_stamp"]
    assert not result[0]["detections"][1]["has_stamp"]
    assert not result[0]["detections"][2]["has_stamp"]
    assert not result[0]["detections"][0]["forced"]
    assert not result[0]["detections"][1]["forced"]
    assert result[0]["detections"][2]["forced"]
