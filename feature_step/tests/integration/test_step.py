import unittest
from unittest import mock
from apf.producers import GenericProducer
from apf.consumers import KafkaConsumer
from features.step import FeaturesComputer
from schema import SCHEMA
from tests.data.message_factory import generate_input_batch
from features.core.ztf import ZTFFeatureExtractor
from features.utils.metrics import get_sid

CONSUMER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "PARAMS": {
        "bootstrap.servers": "server",
        "group.id": "group_id",
        "auto.offset.reset": "beginning",
        "enable.partition.eof": False,
    },
    "TOPICS": ["topic"],
    "consume.messages": "1",
    "consume.timeout": "0",
}

PRODUCER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "TOPIC": "test",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": SCHEMA,
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "TOPIC": "test-scribe",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": SCHEMA,
}


class TestZTFStep(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
        }
        self.step = FeaturesComputer(
            ZTFFeatureExtractor,
            config=self.step_config,
        )
        self.step.scribe_producer = mock.create_autospec(GenericProducer)
        self.step.scribe_producer.produce = mock.MagicMock()

    def test_execute(self):
        input_messages = generate_input_batch(5)
        result = self.step.execute(input_messages)
        self.assertEqual(len(result), 5)
        for output_message in result:
            self.assertIn("aid", output_message)
            self.assertIn("meanra", output_message)
            self.assertIn("meandec", output_message)
            self.assertIn("detections", output_message)
            self.assertIn("non_detections", output_message)
            self.assertIn("xmatches", output_message)
            self.assertIn("features", output_message)
            self.assertEqual(len(output_message["features"]), 178)
        self.step.scribe_producer.produce.assert_called()
        scribe_producer_call_count = self.step.scribe_producer.produce.call_count
        self.assertEqual(scribe_producer_call_count, 5)

    def test_step(self):
        input_messages = generate_input_batch(5)
        consumer: mock.MagicMock = self.step.consumer
        producer: mock.MagicMock = self.step.producer
        consumer.mock_add_spec(KafkaConsumer)
        producer.mock_add_spec(GenericProducer)
        consumer.consume.return_value = input_messages
        self.step.start()
        self.step.scribe_producer.produce.assert_called()
        scribe_producer_call_count = self.step.scribe_producer.produce.call_count
        self.assertEqual(scribe_producer_call_count, 5)
        assert self.step.metrics["sid"] == ["ZTF"]
