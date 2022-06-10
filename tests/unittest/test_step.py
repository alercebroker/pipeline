from unittest import mock, TestCase
from cmirrormaker.utils.consumer import RawKafkaConsumer
from cmirrormaker.utils.producer import RawKafkaProducer
from cmirrormaker.step import CustomMirrormaker
from data.datagen import create_data
from confluent_kafka import Consumer


class StepTestCase(TestCase):
    def setUp(self):
        self.mock_consumer = mock.create_autospec(RawKafkaConsumer)
        self.mock_producer = mock.create_autospec(RawKafkaProducer)
        self.step = CustomMirrormaker(
            consumer=self.mock_consumer,
            config={"PRODUCER_CONFIG": "cmirrormaker.utiles.producer"},
            level=69,
            producer=self.mock_producer,
        )

    def tearDown(self):
        del self.step

    def test_step(self):
        data_batch = create_data(10)
        self.step.execute(data_batch)
        self.mock_producer.produce.assert_called()

    @mock.patch('apf.consumers.kafka.Consumer', autospec=True)
    def test_consumer_deserialization(self, mock):
        #mock_kafka_consumer_class = mock.create_autospec(Consumer)
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
        mock_message = create_data(1)[0]
        deserialized_message = consumer._deserialize_message(mock_message)
        assert mock_message == deserialized_message
