from .test_core import GenericProducerTest
from apf.producers import KafkaProducer, KafkaSchemalessProducer
from unittest import mock
import datetime

import fastavro
import os


@mock.patch("apf.producers.kafka.Producer")
class KafkaProducerTest(GenericProducerTest):
    FILE_PATH = os.path.dirname(__file__)
    PRODUCER_SCHEMA_PATH = os.path.join(
        FILE_PATH, "../examples/kafka_producer_schema.avsc"
    )

    def setUp(self) -> None:
        self.params = {
            "PARAMS": {"bootstrap.servers": "kafka1:9092, kafka2:9092"},
            "TOPIC": "test_topic",
            "SCHEMA_PATH": self.PRODUCER_SCHEMA_PATH,
        }

    def test_produce(self, producer_mock):
        producer_mock.reset_mock()
        self.component = KafkaProducer(self.params)
        super().test_produce(use=self.component)
        producer_mock().produce.assert_called()

    def test_produce_with_key(self, producer_mock):
        producer_mock.reset_mock()
        self.component = KafkaProducer(self.params)
        self.component.set_key_field("key")
        super().test_produce(use=self.component)
        print(producer_mock().produce.call_args[1]["key"])
        assert producer_mock().produce.call_args[1]["key"] == "test"

    def test_produce_with_none(self, producer_mock):
        producer_mock.reset_mock()
        self.component = KafkaProducer(self.params)
        self.component.set_key_field(None)
        super().test_produce(use=self.component)
        assert producer_mock().produce.call_args[1]["key"] == None

    def test_topic_strategy(self, _):
        import copy

        now = datetime.datetime.utcnow()
        date_format = "%Y%m%d"
        params = copy.deepcopy(self.params)
        params.update(
            {
                "TOPIC_STRATEGY": {
                    "CLASS": "apf.core.topic_management.DailyTopicStrategy",
                    "PARAMS": {
                        "topic_format": "apf_test_%s",
                        "date_format": date_format,
                        "change_hour": now.hour,
                        "retention_days": 1,
                    },
                },
            }
        )
        params.pop("TOPIC")
        self.component = KafkaProducer(params)
        topic_before = self.component.topic
        self.assertEqual(len(topic_before), 2)
        self.component.produce({"key": "value", "int": 1})
        self.assertEqual(len(self.component.topic), 1)
        self.assertEqual(self.component.topic[0], topic_before[1])
        params["TOPIC_STRATEGY"]["PARAMS"]["retention_days"] = 2
        self.component = KafkaProducer(params)
        topic_before = self.component.topic
        self.assertEqual(len(topic_before), 2)
        self.component.produce({"key": "value", "int": 1})
        self.assertEqual(len(self.component.topic), 2)
        self.assertEqual(self.component.topic, topic_before)


class TestKafkaSchemalessProducer(GenericProducerTest):
    FILE_PATH = os.path.dirname(__file__)
    PRODUCER_SCHEMA_PATH = os.path.join(
        FILE_PATH, "../examples/kafka_producer_schema.avsc"
    )

    def setUp(self) -> None:
        self.params = {
            "PARAMS": {"bootstrap.servers": "kafka1:9092, kafka2:9092"},
            "TOPIC": "test_topic",
            "SCHEMA_PATH": self.PRODUCER_SCHEMA_PATH,
        }

    def test_serialize_message(self):
        producer = KafkaSchemalessProducer(self.params)

        message = {"key": "test", "int": 0}
        out_avro = producer._serialize_message(message)

        expected = b"\x08test\x00"

        assert out_avro == expected

    def test_serialize_bad_message(self):
        producer = KafkaSchemalessProducer(self.params)

        with self.assertRaises(Exception):
            producer._serialize_message({"no_key": "bad_message"})

    def test_serielize_strict(self):
        producer = KafkaSchemalessProducer(self.params)

        with self.assertRaises(Exception):
            message = {"key": "test", "int": "not an int"}
            producer._serialize_message(message)
