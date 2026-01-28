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
        magic, schema_id = unpack(">bI", bytes_io.read(5))
        assert schema_id == 704
        #print(self.schema)
        data = schemaless_reader(bytes_io, self.schema)
        return data
