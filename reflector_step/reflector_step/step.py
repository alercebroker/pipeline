from typing import Any, Literal

from apf.core.step import DefaultProducer, GenericStep
from reflector_step.utils.consumer import RawKafkaConsumerBytes


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

    keep_original_timestamp: bool
    use_message_topic: bool
    survey: Literal["ztf", "lsst"]

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

        isRawKafkaConsumer = type(self.consumer) is RawKafkaConsumerBytes

        if self.survey == "lsst" and not isRawKafkaConsumer:
            self.producer.set_key_function(lsst_partition)

    def produce(self, messages):
        to_produce = [messages] if isinstance(messages, dict) else messages
        count = 0
        for msg in to_produce:
            count += 1
            producer_kwargs = {"flush": count == len(to_produce)}

            isRawKafkaConsumer = type(self.consumer) is RawKafkaConsumerBytes

            if isRawKafkaConsumer:
                if self.survey == "ztf":                
                    if self.keep_original_timestamp:
                        producer_kwargs["timestamp"] = msg["timestamp"] 

                    if self.use_message_topic:
                        self.producer.topic = [msg["topic"]]
            
                    if msg.get("key"):
                        producer_kwargs["key"] = msg["key"]

                self.producer.produce(msg["value"], **producer_kwargs)

            else:

                if not self.keep_original_timestamp and self.survey == "lsst":
                    msg.pop("timestamp", None)

                if self.use_message_topic:
                    self.producer.topic = [msg.get("topic")]

                self.producer.produce(msg, **producer_kwargs)
                
        if not isinstance(self.producer, DefaultProducer):
            self.logger.info(f"Produced {count} messages")

    def execute(self, messages):
        return messages
