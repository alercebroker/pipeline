import logging
from typing import Generator, List
import uuid

from apf.producers.kafka import KafkaProducer
from .test_core import GenericConsumerTest
from apf.consumers.kafka import (
    KafkaJsonConsumer,
    KafkaConsumer,
    KafkaSchemalessConsumer,
)
import unittest
from unittest import mock
from confluent_kafka import KafkaException
from .message_mock import (
    MessageMock,
    MessageJsonMock,
    SchemalessMessageMock,
    SchemalessBadMessageMock,
)
import datetime
import os
import pytest


def consume(num_messages=1):
    messages = [[MessageMock(False)] * num_messages]
    messages.append([MessageMock(True)])
    return messages


@mock.patch("apf.consumers.kafka.Consumer")
class TestKafkaConsumer(GenericConsumerTest):
    def setUp(self) -> None:
        self.params = {
            "TOPICS": ["apf_test"],
            "PARAMS": {
                "bootstrap.servers": "127.0.0.1:9092",
                "group.id": "apf_test",
            },
        }
        self.component = KafkaConsumer(self.params)

    def test_no_topic(self, _):
        params = {
            "PARAMS": {"bootstrap.servers": "127.0.0.1:9092", "group.id": "apf_test"},
        }

        def initialize_consumer(params):
            self.component = KafkaConsumer(params)

        self.assertRaises(Exception, initialize_consumer, params)

    def test_num_messages_timeout(self, mock_consumer):
        mock_consumer().consume.side_effect = consume(num_messages=10)
        opt_params = [
            {"consume.timeout": 10, "consume.messages": 100},
            {"TIMEOUT": 10, "NUM_MESSAGES": 100},
        ]
        for opt_param in opt_params:
            params = {
                "TOPICS": ["apf_test"],
                "PARAMS": {
                    "bootstrap.servers": "127.0.0.1:9092",
                    "group.id": "apf_test",
                },
            }
            params.update(opt_param)
            self.component = KafkaConsumer(params)
            for msj in self.component.consume():
                self.assertIsInstance(msj, list)
                self.assertEqual(len(msj), 10)
                break

    def test_consume(self, mock_consumer):
        self.component = KafkaConsumer(self.params)
        mock_consumer().consume.side_effect = consume(num_messages=1)
        for msj in self.component.consume():
            self.assertIsInstance(msj, dict)
            break

    def test_batch_consume(self, mock_consumer):
        self.component = KafkaConsumer(self.params)
        mock_consumer().consume.side_effect = consume(num_messages=1)
        for msj in self.component.consume(num_messages=10, timeout=5):
            self.assertIsInstance(msj, list)
            # should be equal to available messages even if num_messages is higher
            self.assertEqual(len(msj), 1)
            break

    def test_consume_error(self, mock_consumer):
        self.component = KafkaConsumer(self.params)
        mock_consumer().consume.side_effect = consume(num_messages=0)
        self.assertRaises(Exception, next, self.component.consume())

    def test_commit_error(self, mock_consumer):
        self.component = KafkaConsumer(self.params)
        mock_consumer().commit.side_effect = KafkaException
        with self.assertRaises(KafkaException):
            self.component.commit()

    def test_commit_retry(self, mock_consumer):
        self.component = KafkaConsumer(self.params)
        mock_consumer().commit.side_effect = ([KafkaException] * 4) + [None]
        self.component.commit()


@mock.patch("apf.consumers.kafka.Consumer")
class TestKafkaConsumerDynamicTopic(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime.datetime.utcnow()
        self.tomorrow = self.now + datetime.timedelta(days=1)
        self.date_format = "%Y%m%d"
        self.topic1 = "apf_test_" + self.now.strftime(self.date_format)
        self.topic2 = "apf_test_" + self.tomorrow.strftime(self.date_format)
        self.params = {
            "TOPIC_STRATEGY": {
                "CLASS": "apf.core.topic_management.DailyTopicStrategy",
                "PARAMS": {
                    "topic_format": "apf_test_%s",
                    "date_format": self.date_format,
                    "change_hour": self.now.hour,
                },
            },
            "PARAMS": {
                "bootstrap.servers": "127.0.0.1:9092",
                "group.id": "apf_test",
            },
        }
        self.component = KafkaConsumer(self.params)

    def test_recognizes_dynamic_topic(self, mock_consumer):
        self.component = KafkaConsumer(self.params)
        self.assertTrue(self.component.dynamic_topic)

    def test_creates_correct_topic_strategy_class(self, mock_consumer):
        from apf.core.topic_management import DailyTopicStrategy

        self.component = KafkaConsumer(self.params)
        self.assertTrue(
            isinstance(
                self.component.topic_strategy,
                DailyTopicStrategy,
            )
        )

    def test_subscribes_to_correct_topic_list(self, mock_consumer):
        self.component = KafkaConsumer(self.params)
        self.assertEqual(self.component.topics, [self.topic1, self.topic2])

    def test_detects_new_topic_while_consuming(self, mock_consumer):
        import copy

        mock_consumer().consume.side_effect = consume(num_messages=2)
        params = copy.deepcopy(self.params)
        np1 = self.now.hour + 1 if self.now.hour <= 24 else 0
        params["TOPIC_STRATEGY"]["PARAMS"]["change_hour"] = np1
        self.component = KafkaConsumer(params)
        self.component.topic_strategy.change_hour = self.now.hour
        self.assertEqual(self.component.topics, [self.topic1])
        for _ in self.component.consume():
            self.component.commit()
            break
        self.assertEqual(self.component.topics, [self.topic1, self.topic2])


class TestKafkaJsonConsumer(unittest.TestCase):
    component = KafkaJsonConsumer
    params = {
        "TOPICS": ["apf_test"],
        "PARAMS": {"bootstrap.servers": "127.0.0.1:9092", "group.id": "apf_test"},
    }

    def test_deserialize(self):
        msg = MessageJsonMock()
        consumer = self.component(self.params)
        consumer._deserialize_message(msg)


class TestKafkaSchemalessConsumer(unittest.TestCase):
    FILE_PATH = os.path.dirname(__file__)
    SCHEMALESS_CONSUMER_SCHEMA_PATH = os.path.join(
        FILE_PATH, "../examples/kafka_schemalessconsumer_schema.avsc"
    )
    SCHEMALESS_CONSUMER_BAD_SCHEMA_PATH = os.path.join(
        FILE_PATH, "../examples/kafka_schemalessconsumer_bad_schema.avsc"
    )

    def test_schema_no_path(self):
        params = {
            "TOPICS": ["apf_test"],
            "PARAMS": {
                "bootstrap.servers": "127.0.0.1:9092",
                "group.id": "apf_test",
            },
        }
        with self.assertRaises(Exception):
            KafkaSchemalessConsumer(params)

    def test_shcema_path_to_bad_file(self):
        params = {
            "TOPICS": ["apf_test"],
            "PARAMS": {
                "bootstrap.servers": "127.0.0.1:9092",
                "group.id": "apf_test",
            },
            "SCHEMA_PATH": self.SCHEMALESS_CONSUMER_BAD_SCHEMA_PATH,
        }
        with self.assertRaises(Exception):
            KafkaSchemalessConsumer(params)

    def test_schemaless_deserialize(self):
        schemaless_avro = SchemalessMessageMock(False)
        expected_message = {"key": "llave", "value": 1}

        params = {
            "TOPICS": ["apf_test"],
            "PARAMS": {
                "bootstrap.servers": "127.0.0.1:9092",
                "group.id": "apf_test",
            },
            "SCHEMA_PATH": self.SCHEMALESS_CONSUMER_SCHEMA_PATH,
        }

        consumer = KafkaSchemalessConsumer(params)

        result = consumer._deserialize_message(schemaless_avro)

        self.assertDictEqual(result, expected_message)

    def test_schemaless_deserialize_bad_message(self):
        schemaless_avro = SchemalessBadMessageMock(False)

        params = {
            "TOPICS": ["apf_test"],
            "PARAMS": {
                "bootstrap.servers": "127.0.0.1:9092",
                "group.id": "apf_test",
            },
            "SCHEMA_PATH": self.SCHEMALESS_CONSUMER_SCHEMA_PATH,
        }

        consumer = KafkaSchemalessConsumer(params)

        with self.assertRaises(Exception):
            consumer._deserialize_message(schemaless_avro)


@pytest.fixture
def consumer():
    def initialize_consumer(topic: List[str], extra_config: dict = {}):
        params = {
            "TOPICS": topic,
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
                "group.id": uuid.uuid4().hex,
                "enable.partition.eof": True,
                "auto.offset.reset": "earliest",
            },
        }
        params.update(extra_config)
        return KafkaConsumer(params)

    yield initialize_consumer


def test_consumer_with_offests(consumer, kafka_service, caplog):
    FILE_PATH = os.path.dirname(__file__)
    PRODUCER_SCHEMA_PATH = os.path.join(
        FILE_PATH, "../examples/kafka_producer_schema_woffset.avsc"
    )
    caplog.set_level(logging.DEBUG)
    producer = KafkaProducer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "TOPIC": "offset_tests",
            "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
        }
    )
    producer.produce({"id": 1}, timestamp=10)
    producer.produce({"id": 3}, timestamp=15)
    producer.produce({"id": 2}, timestamp=20)
    producer.produce({"id": 4}, timestamp=30)
    producer.producer.flush()
    kconsumer = consumer(["offset_tests"], {"offsets": {"start": 10, "end": 20}})
    messages = list(kconsumer.consume())
    assert len(messages) == 2
    assert messages[0]["id"] == 1
    assert messages[1]["id"] == 3
    kconsumer = consumer(["offset_tests"], {"offsets": {"start": 20}})
    messages = list(kconsumer.consume())
    assert len(messages) == 2
    assert messages[0]["id"] == 2
    assert messages[1]["id"] == 4
    kconsumer = consumer(["offset_tests"])
    messages = list(kconsumer.consume())
    assert len(messages) == 4
    kconsumer = consumer(["offset_tests"], {"offsets": {"start": 40}})
    messages = list(kconsumer.consume())
    assert len(messages) == 0
    kconsumer = consumer(["offset_tests"], {"offsets": {"start": 1}})
    messages = list(kconsumer.consume())
    assert len(messages) == 4


def test_consumer_with_offests_multi_topic(consumer, kafka_service, caplog):
    FILE_PATH = os.path.dirname(__file__)
    PRODUCER_SCHEMA_PATH = os.path.join(
        FILE_PATH, "../examples/kafka_producer_schema_woffset.avsc"
    )
    caplog.set_level(logging.DEBUG)
    producer1 = KafkaProducer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "TOPIC": "offset_tests_1",
            "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
        }
    )
    producer2 = KafkaProducer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "TOPIC": "offset_tests_2",
            "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
        }
    )
    producer1.produce({"id": 1}, timestamp=10)
    producer1.produce({"id": 3}, timestamp=15)
    producer1.produce({"id": 2}, timestamp=20)
    producer1.produce({"id": 4}, timestamp=30)
    producer1.producer.flush()
    producer2.produce({"id": 1}, timestamp=10)
    producer2.produce({"id": 3}, timestamp=15)
    producer2.produce({"id": 2}, timestamp=20)
    producer2.produce({"id": 4}, timestamp=30)
    producer2.producer.flush()
    kconsumer = consumer(
        ["offset_tests_1", "offset_tests_2"],
        {"offsets": {"start": 10, "end": 20}},
    )
    messages = list(kconsumer.consume())
    assert len(messages) == 4
    kconsumer = consumer(
        ["offset_tests_1", "offset_tests_2"],
        {"offsets": {"start": 10}},
    )
    messages = list(kconsumer.consume())
    assert len(messages) == 8
    kconsumer = consumer(
        ["offset_tests_1", "offset_tests_2"], {"offsets": {"start": 40}}
    )
    messages = list(kconsumer.consume())
    assert len(messages) == 0


def test_consumer_with_commit(consumer, kafka_service, caplog):
    FILE_PATH = os.path.dirname(__file__)
    PRODUCER_SCHEMA_PATH = os.path.join(
        FILE_PATH, "../examples/kafka_producer_schema_woffset.avsc"
    )
    caplog.set_level(logging.DEBUG)
    producer = KafkaProducer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "TOPIC": "offset_tests_commit",
            "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
        }
    )
    producer.produce({"id": 1}, timestamp=10)
    producer.produce({"id": 3}, timestamp=15)
    producer.produce({"id": 2}, timestamp=20)
    producer.produce({"id": 4}, timestamp=30)
    producer.producer.flush()
    group_id = "test_consumer_with_commit"
    kconsumer = consumer(
        ["offset_tests_commit"],
        {
            "offsets": {"start": 10},
            "PARAMS": {
                "group.id": group_id,
                "bootstrap.servers": "localhost:9092",
                "enable.partition.eof": True,
                "auto.offset.reset": "earliest",
            },
            "consume.timeout": 5,
        },
    )
    messages = []
    for msg in kconsumer.consume():
        messages.append(msg)
        kconsumer.commit()

    print("First consumer finished")
    assert len(messages) == 4
    del kconsumer
    producer.produce({"id": 5}, timestamp=40)
    producer.producer.flush()
    kconsumer = consumer(
        ["offset_tests_commit"],
        {
            "offsets": {"start": 10},
            "PARAMS": {
                "group.id": group_id + "_1",
                "bootstrap.servers": "localhost:9092",
                "enable.partition.eof": True,
                "auto.offset.reset": "earliest",
            },
            "consume.timeout": 5,
        },
    )
    print("Second consumer starting")
    messages = list(kconsumer.consume())
    assert len(messages) == 5
