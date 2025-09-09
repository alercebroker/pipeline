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
        self.producer_key = step_args.get("producer_key", None)

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

    def pre_produce(self, result):
        if self.producer_key and self.consumer.__class__.__name__ != "RawKafkaConsumer":
            self.producer.set_key_field(self.producer_key)
        return result

    def execute(self, messages):
        return messages
