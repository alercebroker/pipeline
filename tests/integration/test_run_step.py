import logging
from apf.metrics.prometheus import PrometheusMetrics
import requests
from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import MongoDatabaseCreator
import pytest
import unittest
from unittest import mock
from apf.consumers import KafkaConsumer
from sorting_hat_step import SortingHatStep
from schema import SCHEMA
from tests.unittest.data.batch import generate_alerts_batch
from prometheus_client import start_http_server
import time

DB_CONFIG = {
    "HOST": "localhost",
    "USERNAME": "test_user",
    "PASSWORD": "test_password",
    "PORT": 27017,
    "DATABASE": "test_db",
    "AUTH_SOURCE": "test_db",
}

PRODUCER_CONFIG = {
    "CLASS": "apf.producers.KafkaProducer",
    "TOPIC": "sorting_hat_stream",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": SCHEMA,
}

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "sorting_hat_consumer",
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
        "enable.partition.eof": True,
    },
    "consume.timeout": 10,
    "consume.messages": 1,
    "TOPICS": ["survey_stream"],
}

METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "EXTRA_METRICS": [
        {"key": "candid", "format": lambda x: str(x)},
    ],
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": "localhost:9092",
            "auto.offset.reset": "smallest",
        },
        "TOPIC": "metrics",
        "SCHEMA": {
            "$schema": "http://json-schema.org/draft-07/schema",
            "$id": "http://example.com/example.json",
            "type": "object",
            "title": "The root schema",
            "description": "The root schema comprises the entire JSON document.",
            "default": {},
            "examples": [
                {"timestamp_sent": "2020-09-01", "timestamp_received": "2020-09-01"}
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
        },
    },
}


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
class MongoIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.database = new_DBConnection(MongoDatabaseCreator)
        self.step_config = {
            "DB_CONFIG": DB_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
        }

    def test_step(self):
        """
        Nota: este test hace trampa. Lo correcto sería llamar a
        step.start() con un KafkaConsumer y tener poblado los tópicos
        de ztf y atlas.
        """
        with mock.patch("apf.consumers.KafkaConsumer") as consumer_mock:
            with mock.patch.object(SortingHatStep, "_write_success"):
                step = SortingHatStep(
                    config=self.step_config,
                    db_connection=self.database,
                    level=logging.DEBUG,
                )
                batch = generate_alerts_batch(
                    100, nearest=10
                )  # generate 110 alerts where 10 alerts are near of another alerts
                consumer_mock().consume.return_value = [batch]
                step.start()
        step.producer.producer.flush()
        messages = self.consume_messages()
        assert len(messages) == 110
        for message in messages:
            # TODO add other assertions
            self.assert_message_stamps(message)

    def consume_messages(self):
        config = CONSUMER_CONFIG.copy()
        config["PARAMS"]["group.id"] = "assert"
        config["TOPICS"] = ["sorting_hat_stream"]
        consumer = KafkaConsumer(config)
        messages = []
        for message in consumer.consume():
            assert consumer.messages[0].key().startswith(b"AL")
            messages.append(message)
        return messages

    def assert_message_stamps(self, message: dict):
        assert message["stamps"]["science"] == b"science"
        if message["tid"] == "ZTF":
            assert message["stamps"]["template"] == b"template"
        if message["tid"] == "ATLAS":
            assert message["stamps"]["template"] == None
        assert message["stamps"]["difference"] == b"difference"


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
class PrometheusIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prometheus_metrics = PrometheusMetrics()
        start_http_server(8000)

    def setUp(self):
        self.database = new_DBConnection(MongoDatabaseCreator)
        self.step_config = {
            "PROMETHEUS": True,
            "DB_CONFIG": DB_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
        }

    def test_step(self):
        """
        Nota: este test hace trampa. Lo correcto sería llamar a
        step.start() con un KafkaConsumer y tener poblado los tópicos de ztf y atlas.

        Por ahora para poder actualizar la versión se harán los llamados
        a los métodos necesarios de forma manual.
        """
        with mock.patch("apf.consumers.KafkaConsumer") as consumer_mock:
            with mock.patch.object(SortingHatStep, "_write_success"):
                step = SortingHatStep(
                    config=self.step_config,
                    db_connection=self.database,
                    level=logging.DEBUG,
                    prometheus_metrics=self.prometheus_metrics,
                )
                batch = generate_alerts_batch(
                    100, nearest=10
                )  # generate 110 alerts where 10 alerts are near of another alerts
                consumer_mock().consume.return_value = [batch]
                step.start()
        result = requests.get("http://localhost:8000/metrics")
        self.assertIn("processed_messages_sum 110.0", result.text)
