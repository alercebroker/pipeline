from apf.producers import KafkaProducer


class CustomKafkaProducer(KafkaProducer):
    """Producer that prevents serialization of the message"""
    def __init__(self, config: dict):
        # Added dummy schema to prevent error in initialization
        config.setdefault('SCHEMA', {'name': 'dummy', 'type': 'record', 'fields': []})
        super().__init__(config)

    def _serialize_message(self, message):
        return message.value()
