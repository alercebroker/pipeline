from fastavro.schema import load_schema
from fastavro.validation import validate_many

from ingestion_step.lsst.strategy import LsstData, LsstStrategy

output_schema = load_schema("../schemas/ingestion_step/lsst/output.avsc")


def test_serialize(lsst_parsed_data: LsstData):
    # Only checking that it does not crash
    # TODO: Implement proper test
    serialized = LsstStrategy.serialize(lsst_parsed_data)
    validate_many(serialized, output_schema, raise_errors=True, strict=True)
