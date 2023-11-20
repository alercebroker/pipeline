import unittest
import pytest
from confluent_kafka import Producer
from reflector_step.step import CustomMirrormaker
from reflector_step.utils import RawKafkaConsumer
from tests.unittest.data.datagen import create_messages


PRODUCER_CONFIG = {
    "TOPIC": "test",
    "PARAMS": {"bootstrap.servers": "localhost:9093"},
}

CONSUMER_CONFIG = {
    "CLASS": "reflector_step.utils.RawKafkaConsumer",
    "TOPICS": ["test_topic"],
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "reflector_step_test",
        "auto.offset.reset": "beginning",
        "enable.partition.eof": True,
    },
    "consume.timeout": 1,
    "consume.messages": 5,
}


@pytest.mark.usefixtures("kafka_service")
class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step_config = {
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
        }

    def test_step_runs(self):
        n_messages = 10
        external = Producer({"bootstrap.servers": "localhost:9092"})
        step = CustomMirrormaker(config=self.step_config)

        messages = create_messages(n_messages, "test_topic")
        for msg in messages:
            external.produce(topic=msg.topic(), value=msg.value())
        external.flush()
        step.start()

    def test_keep_timestamp(self):
        n_messages = 10
        external = Producer({"bootstrap.servers": "localhost:9092"})
        step = CustomMirrormaker(
            config=self.step_config,
            keep_original_timestamp=True,
            use_message_topic=False,
        )

        messages = create_messages(n_messages, "test_topic")
        for msg in messages:
            external.produce(
                topic=msg.topic(), value=msg.value(), timestamp=123
            )
        external.flush()
        step.start()
        consumer = RawKafkaConsumer(
            {
                "CLASS": "reflector_step.utils.RawKafkaConsumer",
                "TOPICS": ["test"],
                "PARAMS": {
                    "bootstrap.servers": "localhost:9093",
                    "group.id": "test_consumer",
                    "auto.offset.reset": "beginning",
                    "enable.partition.eof": True,
                },
                "consume.timeout": 0,
                "consume.messages": 1,
            }
        )
        for msg in consumer.consume():
            assert msg.timestamp()[1] == 123
