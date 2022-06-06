from apf.core.step import GenericStep
from apf.producers import KafkaProducer
import logging


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
        self.producer = KafkaProducer(config["PRODUCER_CONFIG"])

    def produce(self, message):
        self.producer.produce(message)

    def execute(self, message):
        self.produce(message)
