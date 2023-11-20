from apf.consumers import KafkaConsumer


class RawKafkaConsumer(KafkaConsumer):
    """Consumer that prevents deserialization of the message"""

    def _deserialize_message(self, message):
        return message

    def _post_process(self, parsed, original_message):
        return parsed
