import pandas as pd
from fastavro.schema import load_schema
from fastavro.validation import validate_many

from ingestion_step.ztf import serializer
from ingestion_step.ztf.strategy import ZtfData, ZtfStrategy

output_schema = load_schema("../schemas/ingestion_step/output.avsc")


def test_serialize(ztf_parsed_data: ZtfData):
    msgs = ZtfStrategy.serialize(ztf_parsed_data)

    expected_keys = ["oid", "measurement_id", "detections", "non_detections"]

    assert all([list(msg.keys()) == expected_keys for msg in msgs])
    assert sum(map(lambda msg: len(msg["detections"]), msgs)) == len(
        ztf_parsed_data["detections"]
    ) + len(ztf_parsed_data["forced_photometries"])

    assert sum(map(lambda msg: len(msg["non_detections"]), msgs)) == len(
        ztf_parsed_data["non_detections"]
    )

    validate_many(msgs, output_schema, raise_errors=True)


def test_serialize_detections(ztf_parsed_data: ZtfData):
    detections = serializer.serialize_detections(
        ztf_parsed_data["detections"], ztf_parsed_data["forced_photometries"]
    )

    assert "extra_fields" in detections
    assert len(detections) == len(ztf_parsed_data["detections"]) + len(
        ztf_parsed_data["forced_photometries"]
    )
    assert len(detections[detections["forced"]]) == len(
        ztf_parsed_data["forced_photometries"]
    ), (
        "Number of detections with `forced = True` should be equal to the "
        "number of forced_photometries."
    )

    assert detections.dtypes["parent_candid"] == pd.Int64Dtype()


def test_serialize_non_detections(ztf_parsed_data: ZtfData):
    non_detections = serializer.serialize_non_detections(
        ztf_parsed_data["non_detections"]
    )

    assert len(non_detections) == len(ztf_parsed_data["non_detections"])
