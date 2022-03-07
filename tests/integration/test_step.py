import pytest
import unittest

from confluent_kafka import Consumer
from tests.unittest.data.messages import generate_alerts_batch
from ingestion_step.step import IngestionStep
from schema import SCHEMA

DB_CONFIG = {
    "HOST": "localhost",
    "USER": "test_user",
    "PASSWORD": "test_password",
    "PORT": 27017,
    "DATABASE": "test_db",
}

PRODUCER_CONFIG = {
    "TOPIC": "ingestion_stream",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": SCHEMA,
}


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
class StepIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    @classmethod
    def tearDownClass(cls):
        cls.step.driver.drop_db()

    def setUp(self):
        self.step.driver.create_db()

    def test_produce(self):
        batch = generate_alerts_batch(100)  # generate 100 generic alerts
        self.step.execute(batch)

        # Create a consumer to verify streamed data
        consumer = Consumer({
            "bootstrap.servers": "localhost:9092",
            "group.id": 'consumer_test'
        })
        consumer.subscribe(["ingestion_stream"])
        total_count = 0
        try:
            while total_count < 100:
                msg = consumer.poll(1.0)
                if msg is None:
                    print("Waiting for message or event/error in poll()")
                    continue
                elif msg.error():
                    print(f"Error {msg.error()}")
                else:
                    # Check for Kafka message
                    print(msg)
                    total_count += 1
        except Exception as e:
            print(e)
            pass
        finally:
            # Leave group and commit final offsets
            consumer.close()
        print(total_count)
