from consolidated_metrics_step.step import ConsolidatedMetricsStep
from tests.data import FakeMetric

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


@pytest.mark.usefixtures("redis_service")
@pytest.mark.usefixtures("kafka_service")
class StepIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.step_config = {"PIPELINE_ORDER": PIPELINE_ORDER}
        cls.faker_metrics = FakeMetric()

    def test_run_step_random_metrics(self):
        fake_metrics = [self.faker_metrics.create_fake_metric() for _ in range(100)]
        step = ConsolidatedMetricsStep(config=self.step_config)
        step.execute(fake_metrics)

    def test_run_step_with_bingo(self):
        metrics = self.faker_metrics.create_fake_metrics_candid("test_candid")
        step = ConsolidatedMetricsStep(config=self.step_config)
        step.execute(metrics)
        # called is bingo and return true, produce metric
