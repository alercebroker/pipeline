from apf.consumers import KafkaConsumer


class CustomKafkaConsumer(KafkaConsumer):
    """Consumer that prevents deserialization of the message"""
    def _deserialize_message(self, message):
        return message
