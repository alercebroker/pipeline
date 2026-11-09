import io
import pathlib
import struct
from unittest import TestCase, mock

import fastavro

from reflector_step.utils.consumer import (
    LsstMultiVersionRegistryConsumer,
    rubin_schema_id,
)

ROOT = pathlib.Path(__file__).parent.parent.parent.parent
V11_1 = ROOT / "schemas/surveys/lsst_v11.1/lsst.v11_1.alert.avsc"
V11_2 = ROOT / "schemas/surveys/lsst_v11.2/lsst.v11_2.alert.avsc"
# Rubin's sample alert shipped with schema 11.2 (alert_packet schema/11/2/sample_data)
SAMPLE_V11_2 = pathlib.Path(__file__).parent / "data/lsst_v11_2_fakeAlert.avro"

V11_2_NEW_DIA_SOURCE_FIELDS = {
    "pixelFlags_spike",
    "templateMjdMean",
    "templateMjdMin",
    "templateMjdMax",
    "templateInstrument",
    "templateVersion",
    "mixedInstruments",
}


def wire(schema_id: int, schema: dict, record: dict) -> bytes:
    """Encode a record in Confluent wire format, as Rubin publishes it."""
    buf = io.BytesIO()
    buf.write(struct.pack(">bI", 0, schema_id))
    fastavro.schemaless_writer(buf, schema, record)
    return buf.getvalue()


def as_v11_1(alert: dict) -> dict:
    def strip(source):
        return {
            k: v
            for k, v in source.items()
            if k not in V11_2_NEW_DIA_SOURCE_FIELDS
        }

    old = dict(alert)
    old["diaSource"] = strip(alert["diaSource"])
    old["prvDiaSources"] = [strip(s) for s in alert["prvDiaSources"]]
    return old


class FakeMessage:
    def __init__(self, value: bytes):
        self._value = value

    def value(self):
        return self._value


def make_consumer(previous=(V11_1,)):
    return LsstMultiVersionRegistryConsumer(
        {
            "TOPICS": ["lsst-alerts-v11"],
            "PARAMS": {"group.id": "test"},
            "SCHEMA_PATH": str(V11_2),
            "PREVIOUS_SCHEMA_PATHS": [str(p) for p in previous],
        }
    )


@mock.patch("apf.consumers.kafka.Consumer", autospec=True)
class TestLsstMultiVersionRegistryConsumer(TestCase):
    def setUp(self):
        with open(SAMPLE_V11_2, "rb") as f:
            self.alert = next(fastavro.reader(f))
        self.v11_1 = fastavro.schema.load_schema(str(V11_1))
        self.v11_2 = fastavro.schema.load_schema(str(V11_2))

    def test_schema_ids_match_rubin_registry(self, _):
        self.assertEqual(rubin_schema_id(self.v11_1), 1101)
        self.assertEqual(rubin_schema_id(self.v11_2), 1102)

    def test_current_version_is_decoded_unchanged(self, _):
        consumer = make_consumer()
        msg = FakeMessage(wire(1102, self.v11_2, self.alert))
        self.assertEqual(consumer._deserialize_message(msg), self.alert)

    def test_previous_version_is_upgraded(self, _):
        consumer = make_consumer()
        msg = FakeMessage(wire(1101, self.v11_1, as_v11_1(self.alert)))
        decoded = consumer._deserialize_message(msg)

        sources = [decoded["diaSource"], *decoded["prvDiaSources"]]
        for source in sources:
            self.assertIsNone(source["pixelFlags_spike"])
            self.assertIsNone(source["templateMjdMean"])
            self.assertIsNone(source["templateMjdMin"])
            self.assertIsNone(source["templateMjdMax"])
            self.assertEqual(source["templateInstrument"], "UNKNOWN")
            self.assertEqual(source["templateVersion"], 0)
            self.assertIs(source["mixedInstruments"], False)

        # Every pre-existing field survives the upgrade untouched
        expected = as_v11_1(self.alert)
        self.assertEqual(as_v11_1(decoded), expected)

    def test_upgraded_message_can_be_produced_with_new_schema(self, _):
        consumer = make_consumer()
        msg = FakeMessage(wire(1101, self.v11_1, as_v11_1(self.alert)))
        decoded = consumer._deserialize_message(msg)

        buf = io.BytesIO()
        fastavro.schemaless_writer(buf, self.v11_2, decoded)
        buf.seek(0)
        self.assertEqual(fastavro.schemaless_reader(buf, self.v11_2), decoded)

    def test_unknown_schema_id_raises(self, _):
        consumer = make_consumer()
        msg = FakeMessage(wire(1199, self.v11_2, self.alert))
        with self.assertRaisesRegex(ValueError, "unknown schema id 1199"):
            consumer._deserialize_message(msg)

    def test_previous_version_rejected_when_not_configured(self, _):
        consumer = make_consumer(previous=())
        msg = FakeMessage(wire(1101, self.v11_1, as_v11_1(self.alert)))
        with self.assertRaisesRegex(ValueError, "unknown schema id 1101"):
            consumer._deserialize_message(msg)

    def test_bad_magic_byte_raises(self, _):
        consumer = make_consumer()
        payload = bytearray(wire(1102, self.v11_2, self.alert))
        payload[0] = 1
        with self.assertRaisesRegex(ValueError, "magic byte"):
            consumer._deserialize_message(FakeMessage(bytes(payload)))
