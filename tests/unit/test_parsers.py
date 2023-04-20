from prv_candidates_step.core.strategy.ztf import ZTFNonDetectionsParser
from prv_candidates_step.core.strategy.ztf import ZTFPreviousDetectionsParser
from fastavro.utils import generate_many
from fastavro.schema import load_schema


def test_prv_detections_parser():
    schema = load_schema("tests/shared/prv_candidate_schema.avsc")
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    result = ZTFPreviousDetectionsParser.parse_to_dict(data[0], "aid", "oid", "parent_candid")
    assert result["aid"] == "aid"
    assert result["oid"] == "oid"
    assert result["extra_fields"]["parent_candid"] == "parent_candid"


def test_non_detections_parser():
    schema = load_schema("tests/shared/prv_candidate_schema.avsc")
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    result = ZTFNonDetectionsParser.parse_non_detection(data[0], "aid", "oid")
    assert result["aid"] == "aid"
    assert result["tid"] == "ZTF"
    assert result["oid"] == "oid"
