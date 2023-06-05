from apf.metrics.prometheus import PrometheusMetrics
import requests
from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import MongoDatabaseCreator
import pytest
import unittest
from unittest import mock
from sorting_hat_step import SortingHatStep
from schema import SCHEMA

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
    "TOPICS": ["survey_stream"],
    "SCHEMA_PATH": "" #al archivo del esquema
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
                "timestamp_sent": {SCHEMA
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
    def setUp(self):
        self.database = new_DBConnection(MongoDatabaseCreator)
        self.step_config = {
            "DB_CONFIG": DB_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
        }

    def test_step(self):
        # Generar mensajes avro sin schema (dinamica o estaticamente sirve)
        avros = []

        # publicar los mensajes generados a kafka consumer topic


        # ejecutar start_step
        step = SortingHatStep(self.step_config)
        step.start()

        # consumir todo lo que esta en el topico de salida y hacer asserts sobre el contenido
        # guardar los mensajes del topico de salida en step_result
        # trampita: podriamos hacer un mock de produce y hacer assert called with.
        step_result = []

        # En realidad solo importa que el step corra efectivamente y que no se pierda ningun mensaje
        # el contenido del output deberia dar lo mismo y la estructura esta dada por el schema
        self.assertEqual(len(step_result), 0)
