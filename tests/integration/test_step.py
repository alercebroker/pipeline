from confluent_kafka import cimpl
from confluent_kafka import Consumer
from consolidated_metrics_step.step import ConsolidatedMetricsStep
from tests.data import FakeMetric

import json
import pytest
import unittest

PIPELINE_ORDER = {
    "EarlyClassifier": None,
    "S3Step": None,
    "WatchlistStep": None,
    "SortingHatStep": {
        "IngestionStep": {"XmatchStep": {"FeaturesComputer": {"LateClassifier": None}}}
    },
}

PRODUCER_CONFIG = {
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "TOPIC": "consolidated-metrics-dev",
}


def create_consumer(topic, identifier, server="localhost:9092"):
    consumer = Consumer(
        {
            "bootstrap.servers": server,
            "group.id": identifier,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])
    return consumer


@pytest.mark.usefixtures("redis_service")
@pytest.mark.usefixtures("kafka_service")
class StepIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.step_config = {
            "PIPELINE_ORDER": PIPELINE_ORDER,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
        }
        cls.faker_metrics = FakeMetric()

    def test_run_step_random_metrics(self):
        fake_metrics = [self.faker_metrics.create_fake_metric() for _ in range(100)]
        step = ConsolidatedMetricsStep(config=self.step_config)
        step.execute(fake_metrics)

    def test_run_step_with_bingo(self):
        metrics = self.faker_metrics.create_fake_metrics_candid("test_candid")
        step = ConsolidatedMetricsStep(config=self.step_config)
        step.execute(metrics)

        consumer = create_consumer("consolidated-metrics-dev", "test_1")

        while True:
            output = consumer.poll(timeout=10)
            if output.error():
                continue
            else:
                break
        consumer.close()
        self.assertIsInstance(output, cimpl.Message)
        output_val = output.value()
        self.assertIsInstance(output_val, bytes)
        decoded_output = json.loads(output_val.decode("utf-8"))
        self.assertEqual(decoded_output["candid"], "test_candid")
