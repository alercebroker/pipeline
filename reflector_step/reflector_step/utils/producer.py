from apf.producers import KafkaSchemalessProducer

class RawKafkaProducer(KafkaSchemalessProducer):
    """Producer that sends raw bytes without serialization"""
    
    def _serialize_message(self, message):
        if isinstance(message, bytes):
            return message
        return super()._serialize_message(message)