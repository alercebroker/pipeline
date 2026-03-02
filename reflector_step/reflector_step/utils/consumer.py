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