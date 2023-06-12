import io
import fastavro
import pytest
import unittest

from confluent_kafka import Consumer, cimpl
from ingestion_step.step import IngestionStep
from tests.unittest.data.messages import generate_alerts_batch
from typing import List
from schema import SCHEMA


DB_CONFIG = {
    "PSQL": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "PORT": 5432,
        "DB_NAME": "postgres",
    },
    "MONGO": {
        "HOST": "localhost",
        "USER": "test_user",
        "PASSWORD": "test_password",
        "PORT": 27017,
        "DATABASE": "test_db",
    },
}

PRODUCER_CONFIG = {
    "TOPIC": "test",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": SCHEMA,
}


def _chunks(lst: List, n: int) -> List:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _deserialize_message(message: cimpl.Message):
    bytes_io = io.BytesIO(message.value())
    reader = fastavro.reader(bytes_io)
    data = reader.next()
    return data


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("psql_service")
@pytest.mark.usefixtures("kafka_service")
class StepIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # this step only for setup db
        cls.step_config = {
            "DB_CONFIG": DB_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "STEP_METADATA": {
                "STEP_ID": "ingestion",
                "STEP_NAME": "ingestion",
                "STEP_VERSION": "test",
                "STEP_COMMENTS": "test version",
            },
        }
        cls.step = IngestionStep(config=cls.step_config)

    # @classmethod
    # def tearDownClass(cls):
    #     cls.step.driver.drop_db()

    def setUp(self):
        self.step.driver.create_db()

    def test_produce_batches(self):
        total_alerts = 100
        total_objects = 10
        chunk_size = 10
        output_topic = "ingestion_test_1"
        batch = generate_alerts_batch(total_alerts, same_objects=total_objects)
        self.step_config["PRODUCER_CONFIG"]["TOPIC"] = output_topic
        step = IngestionStep(config=self.step_config)

        for mini_batch in _chunks(batch, chunk_size):
            step.execute(mini_batch)

        # Create a consumer to verify streamed data
        consumer = Consumer(
            {
                "bootstrap.servers": "localhost:9092",
                "group.id": "consumer_test_1",
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([output_topic])
        output = []
        old_len = -1
        while True:
            msg = consumer.consume(num_messages=10, timeout=1)
            output += msg
            if old_len == len(output):
                break
            old_len = len(output)
        consumer.close()
        deserialized = [_deserialize_message(x) for x in output]
        self.assertIsInstance(output, list)
        self.assertIsInstance(deserialized[0], dict)
        self.assertIsInstance(output[0], cimpl.Message)
        # output between 10 and 100
        self.assertGreaterEqual(len(output), total_objects)
        self.assertGreaterEqual(total_alerts, len(output))

    def test_produce_batch(self):
        total_alerts = 100
        total_objects = 10
        batch = generate_alerts_batch(total_alerts, same_objects=total_objects)
        output_topic = "ingestion_test_2"
        self.step_config["PRODUCER_CONFIG"]["TOPIC"] = output_topic
        step = IngestionStep(config=self.step_config)
        step.execute(batch)

        consumer = Consumer(
            {
                "bootstrap.servers": "localhost:9092",
                "group.id": "consumer_test_2",
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([output_topic])
        output = []
        old_len = -1
        while True:
            msg = consumer.consume(num_messages=10, timeout=1)
            output += msg
            if old_len == len(output):
                break
            old_len = len(output)
        consumer.close()
        deserialized = [_deserialize_message(x) for x in output]
        self.assertIsInstance(output, list)
        self.assertIsInstance(deserialized[0], dict)
        self.assertIsInstance(output[0], cimpl.Message)
        self.assertEqual(len(output), total_alerts)

        one_light_curve = deserialized[0]
        for field in [
            "aid",
            "candid",
            "detections",
            "non_detections",
            "ndet",
            "meanra",
            "meandec",
        ]:
            self.assertIn(field, one_light_curve)

        one_detection = one_light_curve["detections"][0]
        for field in [
            "oid",
            "tid",
            "candid",
            "mjd",
            "fid",
            "ra",
            "e_ra",
            "dec",
            "e_dec",
            "mag",
            "e_mag",
            "isdiffpos",
            "rb",
            "rbversion",
            "extra_fields",
        ]:
            self.assertIn(field, one_detection)

        one_non_detection = one_light_curve["non_detections"][0]
        for field in ["tid", "oid", "mjd", "diffmaglim", "fid"]:
            self.assertIn(field, one_non_detection)
