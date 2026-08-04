from apf.consumers import KafkaConsumer
import io
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
            "key": str(key) if key is not None else None
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
        self.key_field = config.get("producer_key", "objectId")  # unused; kept for parity

    def _post_process(self, parsed, original_message):
        return {
            "value": parsed,                       # raw bytes, copied verbatim
            "timestamp": original_message.timestamp()[1],
            "topic": original_message.topic(),
            "key": original_message.key(),         # preserve the ORIGINAL Kafka key (bytes / None)
        }