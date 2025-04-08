from ingestion_step.core.parser_interface import ParsedData
from ingestion_step.ztf import serializer


def test_serialize_detections(parsed_ztf_data: ParsedData):
    detections = serializer.serialize_detections(
        parsed_ztf_data["detections"], parsed_ztf_data["forced_photometries"]
    )

    assert "extra_fields" in detections
    assert len(detections) == len(parsed_ztf_data["detections"]) + len(
        parsed_ztf_data["forced_photometries"]
    )


def test_serialize_non_detections(parsed_ztf_data: ParsedData):
    non_detections = serializer.serialize_non_detections(
        parsed_ztf_data["non_detections"]
    )

    assert len(non_detections) == len(parsed_ztf_data["non_detections"])


def test_serialize_ztf(parsed_ztf_data: ParsedData):
    msgs = serializer.serialize_ztf(parsed_ztf_data)

    expected_keys = ["oid", "candid", "detections", "non_detections"]

    assert all([list(msg.keys()) == expected_keys for msg in msgs])
    assert sum(map(lambda msg: len(msg["detections"]), msgs)) == len(
        parsed_ztf_data["detections"]
    ) + len(parsed_ztf_data["forced_photometries"])

    assert sum(map(lambda msg: len(msg["non_detections"]), msgs)) == len(
        parsed_ztf_data["non_detections"]
    )
