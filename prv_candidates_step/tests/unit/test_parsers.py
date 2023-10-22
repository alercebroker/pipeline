from prv_candidates_step.core.strategy.ztf import ZTFNonDetectionsParser
from prv_candidates_step.core.strategy.ztf import ZTFPreviousDetectionsParser
from fastavro.utils import generate_many
from fastavro.schema import load_schema
from fastavro.repository.base import SchemaRepositoryError
from prv_candidates_step.core.strategy.ztf import (
    extract_detections_and_non_detections as ztf_extract,
)
from prv_candidates_step.core.strategy.lsst import (
    extract_detections_and_non_detections as elasticc_extract,
)
import pickle
import pytest


@pytest.mark.asyncio
async def test_prv_detections_parser():
    try:
        schema = load_schema("tests/shared/ztf_prv_candidate_schema.avsc")
    except SchemaRepositoryError:
        schema = load_schema(
            "prv_candidates_step/tests/shared/ztf_prv_candidate_schema.avsc"
        )
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    result = await ZTFPreviousDetectionsParser.parse_message(
        data[0],
        {
            "aid": "aid",
            "oid": "oid",
            "candid": "123",
            "extra_fields": {"ef1": 1},
        },
    )
    assert result["oid"] == "oid"
    assert result["aid"] == "aid"
    assert not result["has_stamp"]
    assert result["extra_fields"]["ef1"] == 1
    assert result["parent_candid"] == "123"


@pytest.mark.asyncio
async def test_ztf_extract_detections_and_non_detections():
    try:
        schema = load_schema("tests/shared/ztf_prv_candidate_schema.avsc")
    except SchemaRepositoryError:
        schema = load_schema(
            "prv_candidates_step/tests/shared/ztf_prv_candidate_schema.avsc"
        )
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1, "candid": 123}, data))
    alert = {
        "aid": "aid",
        "oid": "oid",
        "candid": "123",
        "fid": 1,
        "extra_fields": {"ef1": 1, "prv_candidates": pickle.dumps(data)},
    }
    result = await ztf_extract(alert)
    assert len(result["detections"]) == 2
    for res in result["detections"]:
        assert res["extra_fields"]["ef1"] == 1


def test_elasticc_extract_detections_and_non_detections():
    try:
        dia_source_schema = load_schema("tests/shared/elasticc_prv_candidate.avsc")
    except SchemaRepositoryError:
        dia_source_schema = load_schema(
            "prv_candidates_step/tests/shared/elasticc_prv_candidate.avsc"
        )
    try:
        forced_schema = load_schema("tests/shared/elasticc_forced_schema.avsc")
    except SchemaRepositoryError:
        forced_schema = load_schema(
            "prv_candidates_step/tests/shared/elasticc_forced_schema.avsc"
        )
    prv_dia_sources = list(generate_many(dia_source_schema, 1))
    forced_sources = list(generate_many(forced_schema, 1))
    alert = {
        "aid": "aid",
        "oid": "oid",
        "ra": 10,
        "dec": 20,
        "candid": "123",
        "fid": 1,
        "extra_fields": {
            "ef1": 1,
            "prvDiaSources": pickle.dumps(prv_dia_sources),
            "prvDiaForcedSources": pickle.dumps(forced_sources),
        },
    }
    result = elasticc_extract(alert)
    assert len(result["detections"]) == 3
    for res in result["detections"]:
        assert res["extra_fields"]["ef1"] == 1


@pytest.mark.asyncio
async def test_non_detections_parser():
    try:
        schema = load_schema("tests/shared/ztf_prv_candidate_schema.avsc")
    except SchemaRepositoryError:
        schema = load_schema(
            "prv_candidates_step/tests/shared/ztf_prv_candidate_schema.avsc"
        )
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    result = await ZTFNonDetectionsParser.parse_message(
        data[0],
        {
            "oid": "oid",
            "aid": "aid",
            "extra_fields": {"ef1": 1},
            "stamps": {"science": "science", "template": "template"},
        },
    )
    assert result["tid"] == "ZTF"
    assert result["oid"] == "oid"
    assert result["aid"] == "aid"
    assert result.get("extra_fields") is None
    assert result.get("stamps") is None
