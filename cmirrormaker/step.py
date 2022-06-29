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
                "Producer is defined twice. Using PRODUCER_CONFIG"
            )
        self.use_message_topic = True
        if "PRODUCER_CONFIG" in self.config:
            pconfig = self.config["PRODUCER_CONFIG"]
            if "TOPIC" in pconfig or "TOPIC_STRATEGY" in pconfig:
                self.use_message_topic = False
            producer = get_class(
                pconfig.pop("CLASS", "cmirrormaker.utils.RawKafkaProducer")
            )(pconfig)
        if producer is None:
            raise Exception("Kafka producer not configured in settings.py")
        self.producer = producer

    def produce(self, message):
        try:
            self._produce_single_message(message)
        except (AttributeError, TypeError):
            for msg in message:
                self._produce_single_message(msg)

    def _produce_single_message(self, message):
        if self.use_message_topic:
            self.producer.topic = [message.topic()]
        self.producer.produce(message)

    def execute(self, message):
        self.produce(message)
