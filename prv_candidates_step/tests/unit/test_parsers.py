import pathlib
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


def test_prv_detections_parser():
    try:
        schema = load_schema("tests/shared/ztf_prv_candidate_schema.avsc")
    except SchemaRepositoryError:
        schema = load_schema(
            "prv_candidates_step/tests/shared/ztf_prv_candidate_schema.avsc"
        )
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    result = ZTFPreviousDetectionsParser.parse_message(
        data[0],
        {
            "aid": "aid",
            "oid": "oid",
            "candid": "123",
            "extra_fields": {"ef1": 1},
        },
    )
    assert result["oid"] == "oid"
    assert not result["has_stamp"]
    assert "ef1" not in result["extra_fields"]
    assert "distnr" in result["extra_fields"]
    # Other extra fields should be verified as well
    assert result["parent_candid"] == "123"


def test_ztf_extract_detections_and_non_detections():
    try:
        schema = load_schema("tests/shared/ztf_prv_candidate_schema.avsc")
    except SchemaRepositoryError:
        schema = load_schema(
            "prv_candidates_step/tests/shared/ztf_prv_candidate_schema.avsc"
        )
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    alert = {
        "aid": "aid",
        "oid": "oid",
        "candid": "123",
        "fid": 1,
        "extra_fields": {
            "ef1": 1,
            "prv_candidates": pickle.dumps(data),
            "distnr": -555,
        },
    }
    result = ztf_extract(alert)
    assert len(result["detections"]) == 2
    for res in result["detections"]:
        assert (
            "ef1" in res["extra_fields"]
            if res["candid"] == "123"
            else "ef1" not in res["extra_fields"]
        )
        assert (
            res["extra_fields"]["distnr"] == -555
            if res["candid"] == "123"
            else res["extra_fields"]["distnr"] > 0
        )


def test_elasticc_extract_detections_and_non_detections():
    dia_source_schema = load_schema(
        str(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent.parent.parent,
                "schemas/elasticc",
                "elasticc.v0_9_1.diaSource.avsc",
            )
        )
    )
    forced_schema = load_schema(
        str(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent.parent.parent,
                "schemas/elasticc",
                "elasticc.v0_9_1.diaForcedSource.avsc",
            )
        )
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


def test_non_detections_parser():
    schema = load_schema(
        str(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent.parent.parent,
                "schemas/ztf",
                "prv_candidate.avsc",
            )
        )
    )
    data = list(generate_many(schema, 1))
    data = list(map(lambda x: {**x, "fid": 1}, data))
    result = ZTFNonDetectionsParser.parse_message(
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
    assert result.get("extra_fields") is None
    assert result.get("stamps") is None
