from apf.metrics import KafkaMetricsProducer
from consolidated_metrics_step import ConsolidatedMetricsStep
from tests.commons import PIPELINE_ORDER
from tests.data import FakeMetric
from unittest import mock

import unittest

STEP_CONFIG = {
    "PIPELINE_ORDER": PIPELINE_ORDER,
    "EXPIRE_TIME": 60,
    "PRODUCER_CONFIG": {},
}


class ConsolidatedMetricStepTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_producer = mock.create_autospec(KafkaMetricsProducer)
        self.step = ConsolidatedMetricsStep(config=STEP_CONFIG, producer=mock_producer)

    @mock.patch("redis.connection.ConnectionPool.get_connection")
    @mock.patch("redis_om.model.model.FindQuery.execute")
    def test_test_execute(self, mock_query, mock_redis):
        faker_metrics = FakeMetric()
        batch = [faker_metrics.create_fake_metric(candid=str(c)) for c in range(0, 10)]
        self.step.execute(batch)
