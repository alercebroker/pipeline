from apf.metrics.prometheus import PrometheusMetrics
import requests
import pytest
import unittest
from unittest import mock
from apf.consumers import KafkaConsumer
from sorting_hat_step.step import SortingHatStep
from schemas.output_schema import SCHEMA
from ..unittest.data.batch import generate_alerts_batch
from prometheus_client import start_http_server
from db_plugins.db.mongo._connection import MongoConnection
import pandas as pd

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
    @classmethod
    def setUpClass(cls):
        cls.database = MongoConnection(DB_CONFIG)

    def setUp(self):
        self.step_config = {
            "DB_CONFIG": DB_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
            "RUN_CONESEARCH": "True",
        }

    def tearDown(self):
        self.database.database["object"].delete_many({})

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

        # Check that there are no duplicates inserted
        cursor = self.database.database["object"].find()
        inserted_objects = list(cursor)
        unique_count = pd.DataFrame(messages).aid.nunique()
        assert len(inserted_objects) == unique_count

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
        # LSST has not stamps
        if message["tid"] == "LSST":
            assert message["stamps"]["science"] is None
            assert message["stamps"]["template"] is None
            assert message["stamps"]["difference"] is None
            return

        assert message["stamps"]["science"] == b"science"
        if message["tid"] == "ZTF":
            assert message["stamps"]["template"] == b"template"
        if message["tid"] == "ATLAS":
            assert message["stamps"]["template"] == None
        assert message["stamps"]["difference"] == b"difference"

    def test_write_object(self):
        step = SortingHatStep(
            config=self.step_config,
            db_connection=self.database,
        )

        # Insert a preexisting entry
        self.database.database["object"].insert_one({"_id": 2, "oid": [50, 100]})

        alerts = pd.DataFrame(
            [
                {"oid": 10, "aid": 0, "extra": "extra1"},
                {"oid": 20, "aid": 1, "extra": "extra2"},
                {"oid": 30, "aid": 0, "extra": "extra3"},
                {"oid": 40, "aid": 2, "extra": "extra4"},
                {"oid": 50, "aid": 2, "extra": "extra4"},
            ]
        )
        step.post_execute(alerts)

        cursor = self.database.database["object"].find()
        result = list(cursor)
        expected = [
            {"_id": 2, "oid": [50, 100, 40]},
            {"_id": 0, "oid": [10, 30]},
            {"_id": 1, "oid": [20]},
        ]
        assert len(result) == len(expected) and all(x in result for x in expected)


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
class PrometheusIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prometheus_metrics = PrometheusMetrics()
        start_http_server(8000)
        cls.database = MongoConnection(DB_CONFIG)

    def setUp(self):
        self.step_config = {
            "PROMETHEUS": True,
            "DB_CONFIG": DB_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
            "RUN_CONESEARCH": "True",
        }

    def tearDown(self):
        self.database.database["object"].delete_many({})

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
                    prometheus_metrics=self.prometheus_metrics,
                )
                batch = generate_alerts_batch(
                    100, nearest=10
                )  # generate 110 alerts where 10 alerts are near of another alerts
                consumer_mock().consume.return_value = [batch]
                step.start()
        result = requests.get("http://localhost:8000/metrics")
        self.assertIn("processed_messages_sum 110.0", result.text)
