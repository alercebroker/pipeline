from consolidated_metrics_step.step import ConsolidatedMetricsStep
from tests.data import FakeMetric

import pytest
import unittest


@pytest.mark.usefixtures("redis_service")
@pytest.mark.usefixtures("kafka_service")
class StepIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.step_config = {}
        cls.faker_metrics = FakeMetric()

    def test_run_step(self):
        fake_metrics = [self.faker_metrics.create_fake_metric() for _ in range(100)]
        step = ConsolidatedMetricsStep(config=self.step_config)
        step.execute(fake_metrics)
