import logging

from apf.core import get_class
from apf.core.step import GenericStep


class CustomMirrormaker(GenericStep):
    """CustomMirrormaker Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    config : dict
        Configuration dictionary.
    level : int
        Logging level.
    """

    def __init__(self, consumer=None, config=None, level=logging.INFO):
        super().__init__(consumer, config=config, level=level)
        if 'PRODUCER_CONFIG' not in config:
            raise Exception("Kafka producer not configured in settings.py")

        producer = get_class(config['PRODUCER_CONFIG'].pop('CLASS', 'cmirrormaker.utils.CustomKafkaProducer'))
        self.producer = producer(config['PRODUCER_CONFIG'])

    def produce(self, messages):
        try:
            self.producer.produce(messages)
        except TypeError:
            for message in messages:
                self.producer.produce(message)

    def execute(self, message):
        self.produce(message)
