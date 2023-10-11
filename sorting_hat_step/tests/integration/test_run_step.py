import logging
import os
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
from db_plugins.db.mongo._connection import MongoConnection as DBPMongoConnection
from db_plugins.db.sql._connection import PsqlDatabase as DBPPsqlConnection
from sorting_hat_step.database import MongoConnection, PsqlConnection
import pandas as pd
from sqlalchemy import text

os.environ["LOGGING_DEBUG"] = "True"

MONGO_CONFIG = {
    "host": "localhost",
    "username": "test_user",
    "password": "test_password",
    "port": 27017,
    "database": "test_db",
    "authSource": "test_db",
}

PSQL_CONFIG = {
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
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


@pytest.fixture()
def logging_debug(caplog):
    caplog.set_level(logging.DEBUG)


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
@pytest.mark.usefixtures("psql_service")
@pytest.mark.usefixtures("logging_debug")
class DbIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MONGO_CONFIG["database"] = "test_db"
        cls.mongo_database = DBPMongoConnection(MONGO_CONFIG)
        cls.psql_database = DBPPsqlConnection(PSQL_CONFIG)

    def setUp(self):
        self.step_config = {
            "MONGO_CONFIG": MONGO_CONFIG,
            "PSQL_CONFIG": PSQL_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
            "RUN_CONESEARCH": "True",
            "USE_PSQL": "True",
        }
        self.psql_database.create_db()

    def tearDown(self):
        self.mongo_database.database["object"].delete_many({})
        self.psql_database.drop_db()

    def test_step(self):
        """
        Nota: este test hace trampa. Lo correcto sería llamar a
        step.start() con un KafkaConsumer y tener poblado los tópicos
        de ztf y atlas.
        """
        with mock.patch("apf.consumers.KafkaConsumer") as consumer_mock:
            with mock.patch.object(SortingHatStep, "_write_success"):
                mongo_db = MongoConnection(MONGO_CONFIG)
                psql_db = PsqlConnection(PSQL_CONFIG)
                step = SortingHatStep(
                    config=self.step_config,
                    mongo_connection=mongo_db,
                )
                step.set_psql_driver(psql_db)
                batch = generate_alerts_batch(
                    100, nearest=10
                )  # generate 110 alerts where 10 alerts are near of another alerts
                consumer_mock().consume.return_value = [batch]
                step.start()
        step.producer.producer.flush()
        messages = self.consume_messages()
        assert len(messages) == 110
        for message in messages:
            self.assert_message_stamps(message)
            self.assert_message_timestamps(message)

        # Check that there are no duplicates inserted
        cursor = self.mongo_database.database["object"].find()
        inserted_objects = list(cursor)
        unique_count = pd.DataFrame(messages).aid.nunique()
        assert len(inserted_objects) == unique_count
        unique_count = pd.DataFrame(messages)
        unique_count = unique_count[unique_count.sid == "ZTF"].oid.explode().nunique()
        with self.psql_database.session() as session:
            result = session.execute(text("SELECT * FROM object"))
            result = list(result)
        assert len(result) == unique_count

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

    def assert_message_timestamps(self, message: dict):
        assert message.get("extra_fields").get("brokerIngestTimestamp") is not None
        assert message.get("extra_fields").get("surveyPublishTimestamp") is not None
        assert (
            message["extra_fields"]["surveyPublishTimestamp"]
            <= message["extra_fields"]["brokerIngestTimestamp"]
        )


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
@pytest.mark.usefixtures("psql_service")
@pytest.mark.usefixtures("logging_debug")
class PrometheusIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prometheus_metrics = PrometheusMetrics()
        start_http_server(8000)
        MONGO_CONFIG["database"] = "test_db"
        cls.mongo_database = DBPMongoConnection(MONGO_CONFIG)
        cls.psql_database = DBPPsqlConnection(PSQL_CONFIG)

    def setUp(self):
        self.step_config = {
            "PROMETHEUS": True,
            "MONGO_CONFIG": MONGO_CONFIG,
            "PSQL_CONFIG": PSQL_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
            "RUN_CONESEARCH": "True",
            "USE_PSQL": "True",
        }
        self.psql_database.create_db()

    def tearDown(self):
        self.mongo_database.database["object"].delete_many({})
        self.psql_database.drop_db()

    def test_step(self):
        """
        Nota: este test hace trampa. Lo correcto sería llamar a
        step.start() con un KafkaConsumer y tener poblado los tópicos de ztf y atlas.

        Por ahora para poder actualizar la versión se harán los llamados
        a los métodos necesarios de forma manual.
        """
        with mock.patch("apf.consumers.KafkaConsumer") as consumer_mock:
            with mock.patch.object(SortingHatStep, "_write_success"):
                mongo_db = MongoConnection(MONGO_CONFIG)
                psql_db = PsqlConnection(PSQL_CONFIG)
                step = SortingHatStep(
                    config=self.step_config,
                    mongo_connection=mongo_db,
                    prometheus_metrics=self.prometheus_metrics,
                )
                step.set_psql_driver(psql_db)
                batch = generate_alerts_batch(
                    100, nearest=10
                )  # generate 110 alerts where 10 alerts are near of another alerts
                consumer_mock().consume.return_value = [batch]
                step.start()
        result = requests.get("http://localhost:8000/metrics")
        self.assertIn("processed_messages_sum 110.0", result.text)


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
@pytest.mark.usefixtures("logging_debug")
class OnlyMongoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MONGO_CONFIG["database"] = "test_db"
        cls.mongo_database = DBPMongoConnection(MONGO_CONFIG)

    def setUp(self):
        self.step_config = {
            "PROMETHEUS": False,
            "MONGO_CONFIG": MONGO_CONFIG,
            "PSQL_CONFIG": PSQL_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
            "RUN_CONESEARCH": "True",
            "USE_PSQL": "False",
        }

    def tearDown(self):
        self.mongo_database.database["object"].delete_many({})

    def test_step(self):
        """
        Nota: este test hace trampa. Lo correcto sería llamar a
        step.start() con un KafkaConsumer y tener poblado los tópicos de ztf y atlas.

        Por ahora para poder actualizar la versión se harán los llamados
        a los métodos necesarios de forma manual.
        """
        with mock.patch("apf.consumers.KafkaConsumer") as consumer_mock:
            with mock.patch.object(SortingHatStep, "_write_success"):
                mongo_db = MongoConnection(MONGO_CONFIG)
                step = SortingHatStep(
                    config=self.step_config,
                    mongo_connection=mongo_db,
                )
                batch = generate_alerts_batch(
                    100, nearest=10
                )  # generate 110 alerts where 10 alerts are near of another alerts
                consumer_mock().consume.return_value = [batch]
                step.start()
        result = requests.get("http://localhost:8000/metrics")
        self.assertIn("processed_messages_sum 110.0", result.text)
