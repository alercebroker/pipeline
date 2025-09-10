from typing import Any, Literal

from apf.core.step import DefaultProducer, GenericStep


def lsst_partition(msg: dict[str, Any]) -> str:
    diaObjectId = msg["diaSource"]["diaObjectId"]
    ssObjectId = msg["diaSource"]["ssObjectId"]

    key = ssObjectId
    if key == 0 or key is None:
        key = diaObjectId

    return str(key)


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
        self,
        config: dict[str, Any],
        survey: Literal["ztf", "lsst"],
        keep_original_timestamp: bool = False,
        use_message_topic: bool = False,
        producer_key: str = None,
        **step_args,
    ):
        super().__init__(config=config)
        self.keep_original_timestamp = keep_original_timestamp
        self.use_message_topic = use_message_topic
        self.survey = survey

        isRawKafkaConsumer = self.consumer.__class__.__name__ != "RawKafkaConsumer"

        if self.survey == "lsst" and isRawKafkaConsumer:
            self.producer.set_key_function(lsst_partition)
        elif self.survey == "ztf" and isRawKafkaConsumer:
            self.producer.set_key_field(producer_key)

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
            msg.pop("timestamp")
            self.producer.produce(msg, **producer_kwargs)
        if not isinstance(self.producer, DefaultProducer):
            self.logger.info(f"Produced {count} messages")

    def execute(self, messages):
        return messages
