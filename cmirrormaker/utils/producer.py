from apf.producers import KafkaProducer


class CustomKafkaProducer(KafkaProducer):
    """Producer that prevents serialization of the message"""
    def __init__(self, config: dict):
        # TODO: Added dummy schema to prevent error in initializaiton (unused in production)
        config.setdefault('SCHEMA', {'name': 'dummy', 'type': 'record', 'fields': []})
        super().__init__(config)

    def produce(self, message=None, **kwargs):
        # TODO: Suggest moving serialization in KafkaProducer to own method (as in KafkaConsumer)
        if self.dynamic_topic:
            self.topic = self.topic_strategy.get_topics()
        for topic in self.topic:
            try:
                self.producer.produce(topic, value=message, **kwargs)
                self.producer.poll(0)
            except BufferError as e:
                self.logger.info(f"Error producing message: {e}")
                self.logger.info("Calling poll to empty queue and producing again")
                self.producer.flush()
                self.producer.produce(topic, value=message, **kwargs)
