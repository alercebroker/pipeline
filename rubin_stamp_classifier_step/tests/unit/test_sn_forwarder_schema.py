"""The SN forwarder emits only the keys the forwarder's schema declares (spec 2026-10-29, criteria 1-3)."""
# stdlib: in-memory buffer for the strict writer
import io
# stdlib: path handling for the schema location
import os

# fastavro writes and parses the Avro schema
import fastavro
# schema loader
import fastavro.schema
# test framework
import pytest

# reuse the step factory and topic name of the existing forwarding tests
from tests.unit.test_sn_forwarding import make_step, FORWARD_TOPIC

# path of the real LSST v11.1 alert schema the forwarder uses
LSST_SCHEMA_PATH = os.path.join(  # join the pieces below into one path
    os.path.dirname(__file__), "..", "..", "..", "schemas", "surveys", "lsst_v11.1", "lsst.v11_1.alert.avsc"  # from this test dir up to the repo schemas
)  # end of the path join


def minimal_record(schema, named=None):
    """Smallest value fastavro accepts for a parsed schema: declared defaults, zero values elsewhere."""
    named = named if named is not None else {}  # registry of named types seen so far
    if isinstance(schema, list):  # union: null when allowed, else the first branch
        return None if "null" in schema else minimal_record(schema[0], named)
    if isinstance(schema, str):  # primitive name or reference to a named type
        if schema in named:  # reference to a record already defined
            return minimal_record(named[schema], named)
        return {"null": None, "boolean": False, "int": 0, "long": 0, "float": 0.0,
                "double": 0.0, "bytes": b"", "string": ""}[schema]  # zero value per primitive
    kind = schema["type"]  # complex type tag
    if kind == "record":  # record: default per field or a recursive minimal value
        named[schema["name"]] = schema  # remember it for later references
        return {f["name"]: (f["default"] if "default" in f else minimal_record(f["type"], named))  # default or recursive minimal value per field
                for f in schema["fields"]}  # iterate over the record fields
    if kind == "array":  # empty array is valid
        return []
    if kind == "map":  # empty map is valid
        return {}
    if kind == "enum":  # first symbol is valid
        return schema["symbols"][0]
    if kind == "fixed":  # zero bytes of the declared size
        return b"\x00" * schema["size"]
    return minimal_record(kind, named)  # wrapped primitive such as {"type": "string"}


@pytest.fixture
def lsst_schema():
    """The parsed LSST v11.1 alert schema."""
    return fastavro.schema.load_schema(LSST_SCHEMA_PATH)  # parse the schema file


@pytest.fixture
def schema_step():
    """A step whose forwarder producer is a recording producer loaded with the LSST schema."""
    return make_step(forward_config={  # build the step with a forwarder config
        "CLASS": "tests.unit.recording_producer.RecordingProducer",  # producer that records instead of sending
        "TOPIC": FORWARD_TOPIC,  # forwarder output topic
        "SCHEMA_PATH": LSST_SCHEMA_PATH,  # schema the producer loads, source of sn_forward_fields
    })  # end of the forwarder config


def sn_prediction(alert):
    """A processed message whose ranking-1 class is SN, carrying the given raw alert."""
    return {"oid": 7, "sid": 1, "probabilities": {"SN": 0.9, "AGN": 0.1}, "alert": alert}  # message shaped like the step's processed output


def test_forwarded_alert_with_consumer_timestamp_serializes_strictly(schema_step, lsst_schema):
    """An alert with apf's extra timestamp key is produced in a form the strict writer accepts."""
    alert = dict(minimal_record(lsst_schema), timestamp=1761750000000)  # what apf's consumer hands the step
    schema_step.forward_sn_candidates([sn_prediction(alert)])  # forward it
    (produced, kwargs), = schema_step.sn_forward_producer.produced  # exactly one produced message
    fastavro.schemaless_writer(io.BytesIO(), lsst_schema, produced, strict=True)  # raises before the fix
    assert "timestamp" not in produced  # extra key stripped
    assert kwargs["key"] == "7"  # key is the oid


def test_filter_does_not_mutate_the_alert(schema_step, lsst_schema):
    """Filtering builds a new dict and leaves the original alert untouched."""
    alert = dict(minimal_record(lsst_schema), timestamp=1)  # alert with the extra key
    schema_step.forward_sn_candidates([sn_prediction(alert)])  # forward it
    assert "timestamp" in alert  # original still has it


def test_filter_only_drops_keys(schema_step, lsst_schema):
    """The filter never adds missing schema keys, it only drops unknown ones."""
    alert = {"diaSourceId": 5, "timestamp": 1}  # schema keys missing on purpose
    schema_step.forward_sn_candidates([sn_prediction(alert)])  # forward it
    (produced, _), = schema_step.sn_forward_producer.produced  # exactly one produced message
    assert produced == {"diaSourceId": 5}  # only the unknown key is gone


def test_fields_come_from_the_forwarder_schema_not_the_output_schema(schema_step):
    """sn_forward_fields reflects the forwarder producer's schema, distinct from output_fields."""
    assert "diaSource" in schema_step.sn_forward_fields  # LSST alert field
    assert schema_step.sn_forward_fields != schema_step.output_fields  # not the output topic's fields


def test_producer_without_schema_gets_the_alert_unchanged():
    """With no SCHEMA_PATH the alert is produced as is and no field set is kept."""
    step = make_step(forward_config={"CLASS": "tests.unit.recording_producer.RecordingProducer", "TOPIC": FORWARD_TOPIC})  # forwarder without SCHEMA_PATH
    alert = {"diaSourceId": 5, "timestamp": 1}  # alert with the extra key
    step.forward_sn_candidates([sn_prediction(alert)])  # forward it
    (produced, _), = step.sn_forward_producer.produced  # exactly one produced message
    assert produced == alert  # unchanged
    assert step.sn_forward_fields is None  # no schema, no filter
