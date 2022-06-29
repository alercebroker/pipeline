from unittest import mock, TestCase
from confluent_kafka import Message
from cmirrormaker.utils.consumer import RawKafkaConsumer
from cmirrormaker.utils.producer import RawKafkaProducer
from cmirrormaker.step import CustomMirrormaker
from data.datagen import create_messages


@mock.patch('apf.consumers.kafka.Consumer', autospec=True)
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
                "PARAMS": {}
            }
        )
        mock_message = create_messages()
        deserialized_message = consumer._deserialize_message(mock_message)
        self.assertEqual(mock_message, deserialized_message)


@mock.patch('apf.producers.kafka.Producer', autospec=True)
class TestRawProducer(TestCase):
    def test_serialization_only_returns_message_value(self, mock_producer):
        producer = RawKafkaProducer(
            {
                "TOPIC": None,
                "PARAMS": {}
            }
        )
        message = mock.create_autospec(Message)
        producer._serialize_message(message)
        message.value.assert_called_once()

    def test_initialization_with_schema_logs_warning(self, mock_producer):
        with self.assertLogs(logger='RawKafkaProducer', level='WARNING'):
            RawKafkaProducer(
                {
                    'SCHEMA': {
                        'name': 'dummy',
                        'type': 'record',
                        'fields': []
                    },
                    "TOPIC": None,
                    'PARAMS': {}
                }
            )


class TestStep(TestCase):
    def setUp(self):
        self.mock_consumer = mock.create_autospec(RawKafkaConsumer)
        self.mock_producer = mock.create_autospec(RawKafkaProducer)
        self.step = CustomMirrormaker(
            consumer=self.mock_consumer,
            producer=self.mock_producer,
        )

    def tearDown(self):
        del self.step

    def test_step_fails_if_producer_is_not_defined_in_config(self):
        with self.assertRaisesRegex(Exception, 'producer not configured'):
            CustomMirrormaker(consumer=self.mock_consumer)

    @mock.patch('cmirrormaker.step.get_class')
    def test_step_uses_config_if_producer_is_defined_as_arg_and_in_config(self, mock_class_getter):
        step = CustomMirrormaker(
            consumer=self.mock_consumer,
            producer=self.mock_producer,
            config={
                "PRODUCER_CONFIG": {"TOPIC": None}
            }
        )
        self.assertEqual(mock_class_getter.return_value.return_value, step.producer)

    def test_step_produces_message_batch(self):
        data_batch = create_messages(10)
        self.step.execute(data_batch)
        self.mock_producer.produce.assert_called_with(data_batch[-1])

    def test_step_produces_single_message(self):
        data_batch, = create_messages(1)
        self.step.execute(data_batch)
        self.mock_producer.produce.assert_called_with(data_batch)
