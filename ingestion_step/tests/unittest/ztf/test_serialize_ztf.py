import pandas as pd
from fastavro.schema import load_schema
from fastavro.validation import validate_many

from ingestion_step.ztf import serializer
from ingestion_step.ztf.strategy import ZtfData, ZtfStrategy

output_schema = load_schema(
    "../schemas/ingestion_step/ztf/test_no_extra_fields/output.avsc"
)


def test_serialize(ztf_parsed_data: ZtfData):
    msgs = ZtfStrategy.serialize(ztf_parsed_data)

    expected_keys = [
        "oid",
        "measurement_id",
        "detections",
        "prv_detections",
        "forced_photometries",
        "non_detections",
    ]

    for msg in msgs:
        assert set(msg.keys()) == set(expected_keys)

    for key in expected_keys[2:]:
        total_length = 0
        for msg in msgs:
            total_length += len(msg[key])
        assert total_length == len(ztf_parsed_data[key])

    validate_many(msgs, output_schema, raise_errors=True)


def test_serialize_detections(ztf_parsed_data: ZtfData):
    detections = serializer.serialize_detections(ztf_parsed_data["detections"])
    prv_detections = serializer.serialize_prv_candidates(
        ztf_parsed_data["prv_detections"]
    )
    forced = serializer.serialize_forced_photometries(
        ztf_parsed_data["forced_photometries"]
    )

    assert len(detections) == len(ztf_parsed_data["detections"])
    assert len(prv_detections) == len(ztf_parsed_data["prv_detections"])
    assert len(forced) == len(ztf_parsed_data["forced_photometries"])

    assert not any(detections["forced"])
    assert not any(prv_detections["forced"])
    assert all(forced["forced"])

    assert detections.dtypes["parent_candid"] == pd.Int64Dtype()


def test_serialize_non_detections(ztf_parsed_data: ZtfData):
    non_detections = serializer.serialize_non_detections(
        ztf_parsed_data["non_detections"]
    )

    assert len(non_detections) == len(ztf_parsed_data["non_detections"])
