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

    def __init__(
        self, consumer=None, config=None, level=logging.INFO, producer=None
    ):
        super().__init__(consumer, config=config, level=level)

        if producer and "PRODUCER_CONFIG" in self.config:
            self.logger.warning(
                'Producer is defined twice. Using PRODUCER_CONFIG'
            )
        if "PRODUCER_CONFIG" in self.config:
            producer = get_class(
                self.config["PRODUCER_CONFIG"].pop(
                    "CLASS", "cmirrormaker.utils.RawKafkaProducer"
                )
            )(self.config["PRODUCER_CONFIG"])
        if producer is None:
            raise Exception("Kafka producer not configured in settings.py")
        self.producer = producer

    def produce(self, message):
        try:
            self.producer.produce(message)
        except (AttributeError, TypeError):
            for msg in message:
                self.producer.produce(msg)

    def execute(self, message):
        self.produce(message)
