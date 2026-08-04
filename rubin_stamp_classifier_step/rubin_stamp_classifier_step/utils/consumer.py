from apf.consumers import KafkaConsumer
from struct import unpack
import io
from confluent_kafka import Message
from fastavro import schemaless_reader
from fastavro.schema import load_schema


class LsstKafkaConsumer(KafkaConsumer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.schema = load_schema(config["SCHEMA_PATH"])

    def _deserialize_message(self, message: Message):
        bytes_io = io.BytesIO(message.value())
        # Strip the 5-byte Confluent wire prefix (magic + schema id). The id is
        # Rubin's registry id, which we don't control (we mirror raw bytes), so we
        # don't assert on it; decoding relies on the local SCHEMA_PATH reader schema.
        _magic, _schema_id = unpack(">bI", bytes_io.read(5))
        data = schemaless_reader(bytes_io, self.schema)
        return data
