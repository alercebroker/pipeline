from prv_candidates_step.core.strategy.ztf import ZTFNonDetectionsParser
from prv_candidates_step.core.strategy.ztf import ZTFPreviousDetectionsParser
from fastavro.utils import generate_many
from fastavro.schema import load_schema
from prv_candidates_step.core.strategy.ztf import extract_detections_and_non_detections
import pickle


def test_prv_detections_parser():
    schema = load_schema("tests/shared/ztf_prv_candidate_schema.avsc")
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    result = ZTFPreviousDetectionsParser.parse(data[0], "oid")
    assert result["oid"] == "oid"


def test_extract_detections_and_non_detections():
    schema = load_schema("tests/shared/ztf_prv_candidate_schema.avsc")
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    alert = {
        "aid": "aid",
        "oid": "oid",
        "candid": "123",
        "fid": 1,
        "extra_fields": {"ef1": 1, "prv_candidates": pickle.dumps(data)},
    }
    result = extract_detections_and_non_detections(alert)
    for res in result["detections"]:
        assert res["extra_fields"]["ef1"] == 1


def test_non_detections_parser():
    schema = load_schema("tests/shared/ztf_prv_candidate_schema.avsc")
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    result = ZTFNonDetectionsParser.parse(data[0], "oid")
    assert result["tid"] == "ZTF"
    assert result["oid"] == "oid"
