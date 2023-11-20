from apf.core import get_class
from apf.core.step import GenericStep, DefaultProducer
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

    def __init__(self, config=None, **step_args):
        super().__init__(config=config)
        self.keep_original_timestamp = step_args.get(
            "keep_original_timestamp", False
        )
        self.use_message_topic = step_args.get("use_message_topic", False)

    def produce(self, messages):
        to_produce = [messages] if isinstance(messages, dict) else messages
        count = 0
        for msg in to_produce:
            count += 1
            producer_kwargs = {"flush": count == len(to_produce)}
            if self.keep_original_timestamp:
                producer_kwargs["timestamp"] = msg.timestamp()[1]
            if self.use_message_topic:
                self.producer.topic = [msg.topic()]
            self.producer.produce(msg, **producer_kwargs)
        if not isinstance(self.producer, DefaultProducer):
            self.logger.info(f"Produced {count} messages")

    def execute(self, messages):
        return messages
