import io
import pathlib
import struct
from unittest import TestCase, mock

import fastavro

from reflector_step.utils.consumer import LsstRegistryConsumer, rubin_schema_id

ROOT = pathlib.Path(__file__).parent.parent.parent.parent
V11_1 = ROOT / "schemas/surveys/lsst_v11.1/lsst.v11_1.alert.avsc"
V11_2 = ROOT / "schemas/surveys/lsst_v11.2/lsst.v11_2.alert.avsc"
# Rubin's sample alert shipped with schema 11.2 (alert_packet schema/11/2/sample_data)
SAMPLE_V11_2 = pathlib.Path(__file__).parent / "data/lsst_v11_2_fakeAlert.avro"


def wire(schema_id: int, schema: dict, record: dict, magic: int = 0) -> bytes:
    """Encode a record in Confluent wire format, as Rubin publishes it."""
    buf = io.BytesIO()
    buf.write(struct.pack(">bI", magic, schema_id))
    fastavro.schemaless_writer(buf, schema, record)
    return buf.getvalue()


class FakeMessage:
    def __init__(self, value: bytes):
        self._value = value

    def value(self):
        return self._value

    def topic(self):
        return "lsst-alerts-v11"

    def partition(self):
        return 3

    def offset(self):
        return 42


def make_consumer(schema_path=V11_2):
    return LsstRegistryConsumer(
        {
            "TOPICS": ["lsst-alerts-v11"],
            "PARAMS": {"group.id": "test"},
            "SCHEMA_PATH": str(schema_path),
        }
    )


@mock.patch("apf.consumers.kafka.Consumer", autospec=True)
class TestLsstRegistryConsumer(TestCase):
    def setUp(self):
        with open(SAMPLE_V11_2, "rb") as f:
            self.alert = next(fastavro.reader(f))
        self.v11_1 = fastavro.schema.load_schema(str(V11_1))
        self.v11_2 = fastavro.schema.load_schema(str(V11_2))

    def test_schema_ids_match_rubin_registry(self, _):
        self.assertEqual(rubin_schema_id(self.v11_1), 1101)
        self.assertEqual(rubin_schema_id(self.v11_2), 1102)

    def test_accepts_configured_version(self, _):
        consumer = make_consumer()
        self.assertEqual(consumer.schema_id, 1102)
        decoded = consumer._deserialize_message(
            FakeMessage(wire(1102, self.v11_2, self.alert))
        )
        self.assertEqual(decoded, self.alert)

    def test_decoded_alert_can_be_produced(self, _):
        consumer = make_consumer()
        decoded = consumer._deserialize_message(
            FakeMessage(wire(1102, self.v11_2, self.alert))
        )
        # The reflector re-encodes with KafkaSchemalessProducer (strict mode).
        fastavro.schemaless_writer(
            io.BytesIO(), self.v11_2, decoded, strict=True
        )

    def _as_v11_1(self) -> dict:
        """Rubin's 11.2 sample without the 7 diaSource fields added in 11.2."""
        v11_2_only = {
            "pixelFlags_spike",
            "templateMjdMean",
            "templateMjdMin",
            "templateMjdMax",
            "templateInstrument",
            "templateVersion",
            "mixedInstruments",
        }
        old = dict(self.alert)
        old["diaSource"] = {
            k: v
            for k, v in self.alert["diaSource"].items()
            if k not in v11_2_only
        }
        old["prvDiaSources"] = None
        return old

    def test_rejects_previous_version(self, _):
        consumer = make_consumer()
        with self.assertRaises(ValueError) as ctx:
            consumer._deserialize_message(
                FakeMessage(wire(1101, self.v11_1, self._as_v11_1()))
            )
        message = str(ctx.exception)
        self.assertIn("schema id 1101 (v11.1)", message)
        self.assertIn("only accepts id 1102 (v11.2", message)
        self.assertIn("topic lsst-alerts-v11 partition 3 offset 42", message)

    def test_rejects_newer_and_unrelated_versions(self, _):
        consumer = make_consumer()
        for schema_id in (1103, 1200, 704):
            with self.subTest(schema_id=schema_id):
                with self.assertRaisesRegex(
                    ValueError, f"schema id {schema_id} "
                ):
                    consumer._deserialize_message(
                        FakeMessage(wire(schema_id, self.v11_2, self.alert))
                    )

    def test_rejects_bad_magic_byte(self, _):
        consumer = make_consumer()
        with self.assertRaisesRegex(ValueError, "magic byte 1"):
            consumer._deserialize_message(
                FakeMessage(wire(1102, self.v11_2, self.alert, magic=1))
            )

    def test_rejects_message_without_header(self, _):
        consumer = make_consumer()
        for value in (b"", b"\x00\x00", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "too short"):
                    consumer._deserialize_message(FakeMessage(value))

    def test_rejects_body_longer_than_schema(self, _):
        # Same id, but the bytes don't match our copy of the schema (e.g. Rubin
        # edited 11.2 in place): decoding leaves bytes unread.
        consumer = make_consumer()
        payload = wire(1102, self.v11_2, self.alert) + b"\x00\x02"
        with self.assertRaisesRegex(ValueError, "2 bytes left"):
            consumer._deserialize_message(FakeMessage(payload))

    def test_rejects_body_from_another_schema_with_location(self, _):
        # Same id, but the body was written with another schema (here v11.1):
        # fastavro fails mid-decode instead of leaving bytes over, and the
        # error must still say where the alert is.
        consumer = make_consumer()
        payload = wire(1102, self.v11_1, self._as_v11_1())
        with self.assertRaises(ValueError) as ctx:
            consumer._deserialize_message(FakeMessage(payload))
        message = str(ctx.exception)
        self.assertIn("could not be decoded", message)
        self.assertIn("topic lsst-alerts-v11 partition 3 offset 42", message)
        self.assertIn("differs from SCHEMA_PATH", message)
