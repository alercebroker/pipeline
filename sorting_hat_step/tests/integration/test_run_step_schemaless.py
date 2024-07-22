from apf.consumers import KafkaConsumer
from confluent_kafka import Producer
from .data.messages import generate_schemaless_batch
import pytest
import unittest
from unittest import mock
from sorting_hat_step.step import SortingHatStep
from db_plugins.db.mongo._connection import MongoConnection as DBPMongoConnection
from sorting_hat_step.database import MongoConnection
import os

# SCHEMA PATH RELATIVE TO THE SETTINGS FILE
PRODUCER_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "../../../schemas/sorting_hat_step/output.avsc"
)
METRICS_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "../../../schemas/sorting_hat_step/metrics.json"
)
SCRIBE_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__),
)

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
    "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
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
    "SCHEMA_PATH": os.path.join(
        os.path.dirname(__file__), "../../schemas/elasticc/elasticc.v0_9_1.alert.avscs"
    ),
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
        "SCHEMA_PATH": METRICS_SCHEMA_PATH,
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
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
            "FEATURE_FLAGS": {
                "RUN_CONESEARCH": True,
                "USE_PSQL": False,
                "USE_MONGO": True,
            },
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
        with mock.patch.object(SortingHatStep, "_write_success"):
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
