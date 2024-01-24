import pathlib
from unittest import mock, TestCase
from confluent_kafka import Message
from reflector_step.utils.consumer import RawKafkaConsumer
from reflector_step.utils.producer import RawKafkaProducer
from reflector_step.step import CustomMirrormaker
from data.datagen import create_messages

PATH = pathlib.Path(pathlib.Path(__file__).parent.parent.parent.parent, "schemas/ztf", "alert.avsc")

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
    def test_serialization_only_returns_message_value(self, mock_producer):
        producer = RawKafkaProducer(
            {"TOPIC": None, "PARAMS": {}, "SCHEMA_PATH": PATH}
        )
        message = mock.create_autospec(Message)
        producer._serialize_message(message)
        message.value.assert_called_once()

    def test_initialization_with_schema_logs_warning(self, mock_producer):
        with self.assertLogs(
            logger="alerce.RawKafkaProducer", level="WARNING"
        ):
            RawKafkaProducer(
                {
                    "SCHEMA": {
                        "name": "dummy",
                        "type": "record",
                        "fields": [],
                    },
                    "TOPIC": None,
                    "SCHEMA_PATH": PATH,
                    "PARAMS": {},
                }
            )


class TestStep(TestCase):
    def setUp(self):
        self.settings = {
            "PRODUCER_CONFIG": {
                "CLASS": "unittest.mock.MagicMock",
                "SCHEMA_PATH": PATH,
            },
            "CONSUMER_CONFIG": {
                "CLASS": "unittest.mock.MagicMock",
            },
        }

        self.mock_consumer = mock.create_autospec(RawKafkaConsumer)
        self.mock_producer = mock.create_autospec(RawKafkaProducer)
        self.step = CustomMirrormaker(
            config=self.settings,
            keep_original_timestamp=False,
            use_message_topic=False,
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

    def test_step_with_timestamp(self):
        step = CustomMirrormaker(
            config=self.settings,
            keep_original_timestamp=True,
            use_message_topic=False,
        )
        step.consumer = self.mock_consumer
        step.producer = self.mock_producer

        data_batch = create_messages(10)
        result = step.execute(data_batch)
        step.produce(result)
        self.mock_producer.produce.assert_called_with(
            data_batch[-1], timestamp=123, flush=True
        )
