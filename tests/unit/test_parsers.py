from prv_candidates_step.core.strategy.utils.ztf_non_detections_parser import (
    ZTFNonDetectionsParser,
)
from prv_candidates_step.core.strategy.utils.ztf_prv_detections_parser import (
    ZTFPreviousDetectionsParser,
)
from fastavro.utils import generate_many
from fastavro.schema import load_schema


def test_prv_detections_parser():
    schema = load_schema("tests/shared/prv_candidate_schema.avsc")
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    parser = ZTFPreviousDetectionsParser()
    result = parser.parse(data, "oid", "aid", "parent_candid")
    assert result[0]["aid"] == "aid"
    assert result[0]["oid"] == "oid"
    assert result[0]["parent_candid"] == "parent_candid"


def test_non_detections_parser():
    schema = load_schema("tests/shared/prv_candidate_schema.avsc")
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    parser = ZTFNonDetectionsParser()
    result = parser.parse(data, "aid", "tid", "oid")
    assert result[0]["aid"] == "aid"
    assert result[0]["tid"] == "tid"
    assert result[0]["oid"] == "oid"
