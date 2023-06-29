from apf.consumers import KafkaConsumer


class ArchiveConsumer(KafkaConsumer):
    def __init__(self, config):
        super().__init__(config)

    def _deserialize_message(self, message):
        return message
