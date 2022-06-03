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

    def __init__(self, consumer=None, config=None, producer=None, level=logging.INFO):
        super().__init__(consumer, config=config, level=level)
        self.producer = producer
        if config.get("PRODUCER_CONFIG", False):
            self.producer = KafkaProducer(config["PRODUCER_CONFIG"])

    def produce(self, message):
        pass

    def execute(self, message):
        pass
