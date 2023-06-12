from prv_candidates_step.core.strategy.ztf import ZTFNonDetectionsParser
from prv_candidates_step.core.strategy.ztf import ZTFPreviousDetectionsParser
from fastavro.utils import generate_many
from fastavro.schema import load_schema


def test_prv_detections_parser():
    schema = load_schema("tests/shared/ztf_prv_candidate_schema.avsc")
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    result = ZTFPreviousDetectionsParser.parse(data[0], "oid")
    assert result["oid"] == "oid"


def test_non_detections_parser():
    schema = load_schema("tests/shared/ztf_prv_candidate_schema.avsc")
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    result = ZTFNonDetectionsParser.parse(data[0], "oid")
    assert result["tid"] == "ZTF"
    assert result["oid"] == "oid"
