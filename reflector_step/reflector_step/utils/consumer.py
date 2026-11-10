from apf.consumers import KafkaConsumer
from apf.consumers.kafka import KafkaSchemalessRegistryConsumer
import io
import re
import struct
import fastavro


class RawKafkaConsumer(KafkaConsumer):
    """Consumer that prevents deserialization of the message"""

    def _deserialize_message(self, message):
        return message

    def _post_process(self, parsed, original_message):
        return parsed


class RawKafkaConsumerBytes(KafkaConsumer):
    """
    Consumer that prevents deserialization of the message but returns bytes instead of kafka
    message object used for ztf multisurvey
    """

    def __init__(self, config: dict):
        super().__init__(config)

        schema_path = config.get("SCHEMA_PATH")
        if schema_path:
            self.schema = fastavro.schema.load_schema(schema_path)
        else:
            raise Exception("No Schema path provided")
        self.key_field = config.get("producer_key", "objectId")

    def _deserialize_message(self, message):
        return message.value()

    def _post_process(self, parsed, original_message):
        bytes_io = io.BytesIO(original_message.value())
        reader = fastavro.reader(bytes_io)
        deserialized = next(reader)

        key = deserialized.get(self.key_field)
        return {
            "value": parsed,
            "timestamp": original_message.timestamp()[1],
            "topic": original_message.topic(),
            "key": str(key) if key is not None else None,
        }


class RawKafkaConsumerBytesOriginalKey(RawKafkaConsumerBytes):
    """
    Faithful raw-mirror consumer: copies the message value bytes verbatim and keeps the
    ORIGINAL Kafka message key, WITHOUT deserializing the payload. Use to mirror topics whose
    messages are not (cleanly) Avro-readable (e.g. binary stamp_classifier payloads), where the
    parent's fastavro key-extraction raises UnicodeDecodeError. Needs no SCHEMA_PATH.
    """

    def __init__(self, config: dict):
        # Bypass the parent's mandatory-SCHEMA_PATH __init__; a raw byte copy carries no schema.
        KafkaConsumer.__init__(self, config)
        self.key_field = config.get(
            "producer_key", "objectId"
        )  # unused; kept for parity

    def _post_process(self, parsed, original_message):
        return {
            "value": parsed,  # raw bytes, copied verbatim
            "timestamp": original_message.timestamp()[1],
            "topic": original_message.topic(),
            "key": original_message.key(),  # preserve the ORIGINAL Kafka key (bytes / None)
        }


def rubin_schema_id(schema: dict) -> int:
    """Registry id Rubin assigns to an alert schema.

    Mirrors ``lsst.alert.packet.Schema.get_schema_id``: the two numbers in the
    schema name, minor zero-padded to two digits (``lsst.v11_2.alert`` -> 1102).
    """
    numbers = re.findall(r"\d+", schema["name"])
    if len(numbers) != 2:
        raise ValueError(
            f"Cannot derive a Rubin schema id from schema name {schema['name']!r}"
        )
    return int(numbers[0] + numbers[1].zfill(2))


def _version(schema_id: int) -> str:
    """Inverse of rubin_schema_id: 1102 -> '11.2'."""
    return f"{schema_id // 100}.{schema_id % 100}"


class LsstRegistryConsumer(KafkaSchemalessRegistryConsumer):
    """
    Reads Rubin alerts in Confluent wire format (magic byte + 4-byte schema id +
    Avro body) and only accepts the schema version in ``SCHEMA_PATH``.

    Any other schema id (an older or newer Rubin schema) makes the step fail
    with a clear error instead of decoding the alert with the wrong schema.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.schema_id = rubin_schema_id(self.schema)
        self.logger.info(
            "Accepting only Rubin alerts with schema %s (id %s)",
            self.schema["name"],
            self.schema_id,
        )

    def _deserialize_message(self, message):
        msg_bytes = message.value()
        where = (
            f"topic {message.topic()} partition {message.partition()} "
            f"offset {message.offset()}"
        )
        if msg_bytes is None or len(msg_bytes) < 5:
            raise ValueError(
                f"Rubin alert at {where} is too short to carry a schema id"
            )
        magic, schema_id = struct.unpack(">bI", msg_bytes[:5])
        if magic != 0:
            raise ValueError(
                f"Rubin alert at {where} has magic byte {magic}, expected 0 "
                "(Confluent wire format)"
            )
        if schema_id != self.schema_id:
            raise ValueError(
                f"Rubin alert at {where} uses schema id {schema_id} "
                f"(v{_version(schema_id)}), but this reflector only accepts "
                f"id {self.schema_id} (v{_version(self.schema_id)}, "
                f"{self.schema['name']}). Rubin changed the alert schema: "
                "update SCHEMA_PATH and the pipeline before resuming."
            )

        # Same id but a different schema behind it (e.g. Rubin edited the
        # version in place): fastavro usually fails mid-decode with a bare
        # IndexError/EOFError/etc.; leftover bytes only catch some cases.
        mismatch = (
            f"the schema behind id {schema_id} differs from SCHEMA_PATH "
            f"({self.schema['name']})."
        )
        body = io.BytesIO(msg_bytes)
        body.seek(5)
        try:
            data = fastavro.schemaless_reader(body, self.schema)
        except Exception as e:
            raise ValueError(
                f"Rubin alert at {where} could not be decoded "
                f"({type(e).__name__}: {e}): {mismatch}"
            ) from e
        leftover = len(msg_bytes) - body.tell()
        if leftover:
            raise ValueError(
                f"Rubin alert at {where} has {leftover} bytes left after "
                f"decoding: {mismatch}"
            )
        return data
