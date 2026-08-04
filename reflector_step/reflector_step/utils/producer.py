import json
import os
import tempfile

from apf.producers import KafkaSchemalessProducer

class RawKafkaProducer(KafkaSchemalessProducer):
    """Producer that sends raw bytes without serialization"""

    def _serialize_message(self, message):
        if isinstance(message, bytes):
            return message
        return super()._serialize_message(message)


class RawKafkaProducerNoSchema(RawKafkaProducer):
    """
    RawKafkaProducer that requires NO SCHEMA_PATH. The apf producer base loads a schema in
    __init__ (fastavro.schema.load_schema), but a raw byte passthrough never serializes via a
    schema, so we satisfy that load with a throwaway minimal record schema. Use for faithful
    raw mirroring where no real schema exists / the source schema isn't self-contained.
    Honors an explicit SCHEMA_PATH if one is provided (backward compatible).
    """
    _DUMMY_SCHEMA = {"type": "record", "name": "RawPassthrough", "fields": []}

    def __init__(self, config: dict):
        if not config.get("SCHEMA_PATH"):
            path = os.path.join(tempfile.gettempdir(), "reflector_raw_passthrough.avsc")
            if not os.path.exists(path):
                with open(path, "w") as f:
                    json.dump(self._DUMMY_SCHEMA, f)
            config = {**config, "SCHEMA_PATH": path}
        super().__init__(config)