from apf.producers import KafkaProducer


class RawKafkaProducer(KafkaProducer):
    """Producer that prevents serialization of the message"""
    def __init__(self, config: dict):
        if 'SCHEMA' in config:
            self.logger.warning(f'SCHEMA in PRODUCER_CONFIG defined but is ignored in {self.__class__.__name__}')
        config.setdefault('SCHEMA', {'name': 'dummy', 'type': 'record', 'fields': []})  # Dummy schema
        super().__init__(config)

    def _serialize_message(self, message):
        return message.value()
