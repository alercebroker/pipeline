import io
import json
import pathlib
import tempfile
from unittest import TestCase, mock

import fastavro
from apf.consumers import GenericConsumer
from apf.producers import GenericProducer
from data.datagen import create_messages, create_raw_bytes_messages

from reflector_step.step import CustomMirrormaker
from reflector_step.utils.consumer import (
    RawKafkaConsumer,
    RawKafkaConsumerBytes,
)
from reflector_step.utils.producer import RawKafkaProducer

PATH = pathlib.Path(
    pathlib.Path(__file__).parent.parent.parent.parent,
    "schemas/ztf",
    "alert.avsc",
)

SIMPLE_SCHEMA = {
    "name": "simple",
    "type": "record",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "message", "type": "string"},
    ],
}


@mock.patch("apf.consumers.kafka.Consumer", autospec=True)
class TestRawConsumer(TestCase):
    def test_deserialization_does_not_modify_message(self, mock_consumer):
        consumer = RawKafkaConsumer(
            {
                "TOPIC_STRATEGY": {
                    "CLASS": "apf.core.topic_management.DailyTopicStrategy",
                    "PARAMS": {
                        "topic_format": [
                            "ztf_%s_programid1",
                            "ztf_%s_programid3",
                        ],
                        "date_format": "%Y%m%d",
                        "change_hour": 23,
                        "retention_days": 8,
                    },
                },
                "PARAMS": {},
                "SCHEMA_PATH": PATH,
            }
        )
        mock_message = create_messages()
        deserialized_message = consumer._deserialize_message(mock_message)
        self.assertEqual(mock_message, deserialized_message)


@mock.patch("apf.producers.kafka.Producer", autospec=True)
class TestRawProducer(TestCase):
    def setUp(self):
        self.schema_dir = tempfile.TemporaryDirectory()
        self.schema_path = pathlib.Path(self.schema_dir.name, "schema.avsc")
        self.schema_path.write_text(json.dumps(SIMPLE_SCHEMA))

    def tearDown(self):
        self.schema_dir.cleanup()

    def test_serialization_passes_bytes_through(self, mock_producer):
        producer = RawKafkaProducer(
            {"TOPIC": None, "PARAMS": {}, "SCHEMA_PATH": self.schema_path}
        )
        message = b"\x00raw-avro-payload"
        self.assertIs(producer._serialize_message(message), message)

    def test_serialization_encodes_dict_with_schema(self, mock_producer):
        producer = RawKafkaProducer(
            {"TOPIC": None, "PARAMS": {}, "SCHEMA_PATH": self.schema_path}
        )
        message = {"id": 42, "message": "hello"}
        serialized = producer._serialize_message(message)
        decoded = fastavro.schemaless_reader(
            io.BytesIO(serialized), fastavro.parse_schema(SIMPLE_SCHEMA)
        )
        self.assertEqual(decoded, message)


class TestStep(TestCase):
    def setUp(self):
        self.settings = {
            "PRODUCER_CONFIG": {
                "CLASS": "apf.core.step.DefaultProducer",
                "SCHEMA_PATH": PATH,
            },
            "CONSUMER_CONFIG": {
                "CLASS": "apf.core.step.DefaultConsumer",
            },
        }

        self.mock_consumer = mock.create_autospec(GenericConsumer)
        self.mock_producer = mock.create_autospec(GenericProducer)
        self.step = CustomMirrormaker(
            config=self.settings,
            keep_original_timestamp=False,
            use_message_topic=False,
            survey="ztf",
        )
        self.step.consumer = self.mock_consumer
        self.step.producer = self.mock_producer

    def tearDown(self):
        del self.step

    def test_step_produces_message_batch(self):
        data_batch = create_messages(10)
        result = self.step.execute(data_batch)
        self.step.produce(result)
        self.mock_producer.produce.assert_called_with(
            data_batch[-1], flush=True
        )

    def test_step_produces_single_message(self):
        (data_batch,) = create_messages(1)
        result = self.step.execute(data_batch)
        self.step.produce([result])
        self.mock_producer.produce.assert_called_with(data_batch, flush=True)


class TestStepRawBytesConsumer(TestCase):
    def setUp(self):
        self.settings = {
            "PRODUCER_CONFIG": {
                "CLASS": "apf.core.step.DefaultProducer",
                "SCHEMA_PATH": PATH,
            },
            "CONSUMER_CONFIG": {
                "CLASS": "apf.core.step.DefaultConsumer",
            },
        }
        self.mock_consumer = mock.create_autospec(
            RawKafkaConsumerBytes, instance=True
        )
        self.mock_producer = mock.create_autospec(GenericProducer)

    def create_step(
        self, keep_original_timestamp=False, use_message_topic=False
    ):
        step = CustomMirrormaker(
            config=self.settings,
            keep_original_timestamp=keep_original_timestamp,
            use_message_topic=use_message_topic,
            survey="ztf",
        )
        step.consumer = self.mock_consumer
        step.producer = self.mock_producer
        return step

    def test_step_produces_raw_value(self):
        step = self.create_step()
        data_batch = create_raw_bytes_messages(10)
        step.produce(step.execute(data_batch))
        self.mock_producer.produce.assert_called_with(
            data_batch[-1]["value"], key=data_batch[-1]["key"], flush=True
        )
        self.assertEqual(self.mock_producer.produce.call_count, 10)

    def test_step_with_timestamp(self):
        step = self.create_step(keep_original_timestamp=True)
        data_batch = create_raw_bytes_messages(10)
        step.produce(step.execute(data_batch))
        self.mock_producer.produce.assert_called_with(
            data_batch[-1]["value"],
            timestamp=123,
            key=data_batch[-1]["key"],
            flush=True,
        )

    def test_step_omits_missing_key(self):
        step = self.create_step()
        (message,) = create_raw_bytes_messages(1, key=None)
        step.produce([message])
        self.mock_producer.produce.assert_called_with(
            message["value"], flush=True
        )

    def test_step_uses_message_topic(self):
        step = self.create_step(use_message_topic=True)
        (message,) = create_raw_bytes_messages(
            1, topic="ztf_20261030_programid1"
        )
        step.produce([message])
        self.assertEqual(self.mock_producer.topic, ["ztf_20261030_programid1"])
