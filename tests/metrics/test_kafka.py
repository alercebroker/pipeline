import unittest
from unittest import mock
from apf.metrics.kafka import Producer, KafkaMetricsProducer
import datetime


class KafkaMetricsProducerTest(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "$schema": "http://json-schema.org/draft-07/schema",
            "$id": "http://example.com/example.json",
            "type": "object",
            "title": "The root schema",
            "description": "The root schema comprises the entire JSON document.",
            "default": {},
            "examples": [
                {
                    "timestamp_sent": "2020-09-01",
                    "timestamp_received": "2020-09-01",
                }
            ],
            "required": ["timestamp_sent", "timestamp_received"],
            "properties": {
                "timestamp_sent": {
                    "$id": "#/properties/timestamp_sent",
                    "type": "string",
                    "title": "The timestamp_sent schema",
                    "description": "Timestamp sent refers to the time at which a message is sent.",
                    "default": "",
                    "examples": ["2020-09-01"],
                },
                "timestamp_received": {
                    "$id": "#/properties/timestamp_received",
                    "type": "string",
                    "title": "The timestamp_received schema",
                    "description": "Timestamp received refers to the time at which a message is received.",
                    "default": "",
                    "examples": ["2020-09-01"],
                },
            },
            "additionalProperties": True,
        }
        self.config = {
            "PARAMS": {
                "bootstrap.servers": "fake",
                "auto.offset.reset": "smallest",
            },
            "TOPIC": "test",
            "SCHEMA": self.schema,
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
            "SCHEMA": self.schema,
        }
        producer = KafkaMetricsProducer(self.config, producer=self.mock_producer)
        producer.send_metrics(metrics)
        topic = "test" + self.now.strftime("%Y%m%d")
        self.mock_producer.produce.assert_called_with(topic, parsed_date)
