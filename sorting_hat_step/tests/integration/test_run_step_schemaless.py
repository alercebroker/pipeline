from apf.consumers import KafkaConsumer
from confluent_kafka import Producer
from .data.messages import generate_schemaless_batch
import pytest
import unittest
from unittest import mock
from sorting_hat_step.step import SortingHatStep
from schemas.output_schema import SCHEMA
from schemas.scribe_schema import SCRIBE_SCHEMA
from db_plugins.db.mongo._connection import MongoConnection as DBPMongoConnection
from db_plugins.db.sql._connection import PsqlDatabase as DBPPsqlConnection
from fastavro.repository.base import SchemaRepositoryError
from sorting_hat_step.database import MongoConnection, PsqlConnection

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
    "TOPIC": "sorting_hat_stream_schemaless",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": SCHEMA,
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "apf.producers.KafkaProducer",
    "TOPIC": "w_metadata",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": SCRIBE_SCHEMA
}

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaSchemalessConsumer",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "sorting_hat_consumer",
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
        "enable.partition.eof": True,
    },
    "consume.timeout": 10,
    "consume.messages": 1,
    "TOPICS": ["survey_stream_schemaless"],
    "SCHEMA_PATH": "schemas/elasticc/elasticc.v0_9_1.alert.avscs",
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
class SchemalessConsumeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mongo_database = DBPMongoConnection(MONGO_CONFIG)

    def setUp(self):
        self.step_config = {
            "MONGO_CONFIG": MONGO_CONFIG,
            "PSQL_CONFIG": PSQL_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
            "RUN_CONESEARCH": "True",
            "USE_PSQL": "False",
        }

    def tearDown(self):
        self.mongo_database.database["object"].delete_many({})

    def test_step(self):
        # publicar los mensajes generados a kafka consumer topic
        test_producer_confg = PRODUCER_CONFIG["PARAMS"]
        # test_producer_confg = {"bootstrap.servers": "localhost:9092"}
        producer = Producer(test_producer_confg)
        for message in generate_schemaless_batch(5):
            producer.produce("survey_stream_schemaless", value=message, key=None)

        # ejecutar start_step
        mongo_db = MongoConnection(MONGO_CONFIG)
        psql_db = PsqlConnection(PSQL_CONFIG)
        with mock.patch.object(SortingHatStep, "_write_success"):
            try:
                step = SortingHatStep(
                    mongo_connection=mongo_db,
                    config=self.step_config,
                )
            except SchemaRepositoryError:
                config = self.step_config.copy()
                config["CONSUMER_CONFIG"][
                    "SCHEMA_PATH"
                ] = "sorting_hat_step/schemas/elasticc/elasticc.v0_9_1.alert.avscs"
                step = SortingHatStep(
                    mongo_connection=mongo_db,
                    config=self.step_config,
                )
            step.start()
            step.producer.producer.flush()

        # consumir todo lo que esta en el topico de salida y hacer asserts sobre el contenido
        config = CONSUMER_CONFIG.copy()
        config["PARAMS"]["group.id"] = "assert"
        config["TOPICS"] = ["sorting_hat_stream_schemaless"]
        consumer = KafkaConsumer(config)
        step_result = []
        for message in consumer.consume():
            step_result.append(message)

        self.assertEqual(len(step_result), 5)
