import unittest
from unittest import mock
from apf.metrics.kafka import Producer, KafkaMetricsProducer
import datetime
import os


class KafkaMetricsProducerTest(unittest.TestCase):
    def setUp(self):
        FILE_PATH = os.path.dirname(__file__)
        self.schema_path = os.path.join(
            FILE_PATH, "../examples/kafka_producer_schema.avsc"
        )

        self.config = {
            "PARAMS": {
                "bootstrap.servers": "fake",
                "auto.offset.reset": "smallest",
            },
            "TOPIC": "test",
            "SCHEMA_PATH": self.schema_path,
        }
        self.mock_producer = mock.create_autospec(Producer)
        self.now = datetime.datetime.utcnow()

    def test_send_metrics(self):
        date = datetime.datetime(2020, 9, 18)
        metrics = {"timestamp_received": date, "timestamp_sent": date}
        parsed_date = b'{"timestamp_received": "2020-09-18T00:00:00", "timestamp_sent": "2020-09-18T00:00:00"}'
        producer = KafkaMetricsProducer(self.config, producer=self.mock_producer)
        producer.send_metrics(metrics)
        self.mock_producer.produce.assert_called_with("test", parsed_date)

    def test_send_metrics_topic_strategy(self):
        date = datetime.datetime(2020, 9, 18)
        metrics = {"timestamp_received": date, "timestamp_sent": date}
        parsed_date = b'{"timestamp_received": "2020-09-18T00:00:00", "timestamp_sent": "2020-09-18T00:00:00"}'
        np1 = self.now.hour + 1 if self.now.hour <= 24 else 0
        self.config = {
            "PARAMS": {
                "bootstrap.servers": "fake",
                "auto.offset.reset": "smallest",
            },
            "TOPIC_STRATEGY": {
                "CLASS": "apf.core.topic_management.DailyTopicStrategy",
                "PARAMS": {
                    "topic_format": "test%s",
                    "date_format": "%Y%m%d",
                    "change_hour": np1,
                },
            },
            "SCHEMA_PATH": self.schema_path,
        }
        producer = KafkaMetricsProducer(self.config, producer=self.mock_producer)
        producer.send_metrics(metrics)
        topic = "test" + self.now.strftime("%Y%m%d")
        self.mock_producer.produce.assert_called_with(topic, parsed_date)
