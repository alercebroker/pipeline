from apf.consumers import KafkaConsumer
from apf.consumers.kafka import KafkaSchemalessRegistryConsumer
import copy
import io
import logging
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


class LsstMultiVersionRegistryConsumer(KafkaSchemalessRegistryConsumer):
    """
    Reads Rubin alerts in Confluent wire format (magic byte + 4-byte schema id +
    Avro body) and decodes each message with the schema matching its id, so the
    reflector keeps working across a Rubin schema change without a coordinated
    redeploy.

    Messages written with an older schema are upgraded (Avro schema resolution)
    to ``SCHEMA_PATH``, so everything downstream sees a single schema version.
    Fields added in ``SCHEMA_PATH`` must have a default to be filled for older
    messages; Rubin's required fields without one get the value from
    ``UPGRADE_DEFAULTS``. A message with an unknown schema id raises instead of
    being decoded with the wrong schema.

    Extra config keys:
      PREVIOUS_SCHEMA_PATHS: list of older Rubin alert schemas still accepted.
    """

    # Values for fields Rubin added as required (no Avro default), used only
    # when upgrading older-schema messages. Keyed by unqualified record name.
    UPGRADE_DEFAULTS = {
        "diaSource": {
            # v11.2: Rubin's documented sentinel for "instrument not known".
            "templateInstrument": "UNKNOWN",
            # v11.2: 0/False = template from the same instrument as the
            # diaSource, which holds for every alert before DECam templates.
            "templateVersion": 0,
            "mixedInstruments": False,
        },
    }

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        self.schema_id = rubin_schema_id(self.schema)
        self.writer_schemas = {self.schema_id: self.schema}
        # One reader schema per older version: Avro resolution in fastavro
        # matches named types by full name, so each reader uses the writer's
        # namespace (lsst.v11_1 vs lsst.v11_2). Namespaces don't affect the
        # decoded dict nor the binary encoding.
        self.reader_schemas = {}
        target = self._target_with_upgrade_defaults(config["SCHEMA_PATH"])
        for path in config.get("PREVIOUS_SCHEMA_PATHS") or []:
            schema = fastavro.schema.load_schema(path)
            schema_id = rubin_schema_id(schema)
            self.writer_schemas[schema_id] = schema
            self.reader_schemas[schema_id] = fastavro.parse_schema(
                _rename_namespace(
                    copy.deepcopy(target),
                    _namespace(self.schema),
                    _namespace(schema),
                )
            )
        self._seen_schema_ids = set()
        self.logger.info(
            "Decoding Rubin schema ids %s, upgrading to %s",
            sorted(self.writer_schemas),
            self.schema_id,
        )

    def _target_with_upgrade_defaults(self, path) -> dict:
        """Unparsed target schema with UPGRADE_DEFAULTS as Avro defaults.

        Defaults only affect reading older data; the binary encoding of the
        target schema is unchanged.
        """
        raw = copy.deepcopy(
            fastavro.schema.load_schema(path, _write_hint=False)
        )

        def visit(node):
            if isinstance(node, list):
                for item in node:
                    visit(item)
            elif isinstance(node, dict):
                name = str(node.get("name", "")).split(".")[-1]
                values = self.UPGRADE_DEFAULTS.get(name)
                if node.get("type") == "record" and values:
                    for field in node["fields"]:
                        if field["name"] in values:
                            field.setdefault("default", values[field["name"]])
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        visit(value)

        # UPGRADE_DEFAULTS entries for fields the target schema lacks (e.g. an
        # older SCHEMA_PATH) are simply unused.
        visit(raw)
        return raw

    def _deserialize_message(self, message):
        msg_bytes = message.value()
        magic, schema_id = struct.unpack(">bI", msg_bytes[:5])
        if magic != 0:
            raise ValueError(f"Unexpected Confluent magic byte {magic}")
        writer = self.writer_schemas.get(schema_id)
        if writer is None:
            raise ValueError(
                f"Rubin alert with unknown schema id {schema_id}; known ids: "
                f"{sorted(self.writer_schemas)}. Add its schema to "
                "PREVIOUS_SCHEMA_PATHS or update SCHEMA_PATH."
            )
        if schema_id not in self._seen_schema_ids:
            self._seen_schema_ids.add(schema_id)
            self.logger.info(
                "First message with Rubin schema id %s", schema_id
            )

        body = io.BytesIO(msg_bytes[5:])
        if schema_id == self.schema_id:
            return fastavro.schemaless_reader(body, writer)
        return fastavro.schemaless_reader(
            body, writer, self.reader_schemas[schema_id]
        )


def _namespace(schema: dict) -> str:
    return schema["name"].rsplit(".", 1)[0]


def _rename_namespace(node, old: str, new: str):
    """Rewrite named-type names and references from namespace old to new.

    Only touches schema positions (name/namespace/type/items/values and union
    members), never docs or defaults.
    """

    def fix(name):
        if isinstance(name, str) and name.startswith(old + "."):
            return new + name[len(old) :]
        return name

    if isinstance(node, list):
        return [
            (
                fix(item)
                if isinstance(item, str)
                else _rename_namespace(item, old, new)
            )
            for item in node
        ]
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "namespace" and value == old:
                out[key] = new
            elif key in ("name", "type", "items", "values") and isinstance(
                value, str
            ):
                out[key] = fix(value)
            elif key in ("type", "items", "values", "fields") and isinstance(
                value, (dict, list)
            ):
                out[key] = _rename_namespace(value, old, new)
            else:
                out[key] = value
        return out
    return node
