from consolidated_metrics_step.utils.metric import ConsolidatedMetric
from consolidated_metrics_step.utils.metric import STEP_MAPPER
from consolidated_metrics_step.utils.metric import StepMetric
from datetime import datetime
from tests.commons import PIPELINE_ORDER
from unittest import mock

import unittest


class ConsolidatedMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    @mock.patch("redis.connection.ConnectionPool.get_connection")
    def test_create_metric_model(self, mock_connection):
        cm = ConsolidatedMetric(candid="test", survey="test")
        assert cm.candid == "test"
        assert cm.survey == "test"

    @mock.patch("redis.connection.ConnectionPool.get_connection")
    def test_is_bingo_true(self, mock_connection):
        cm = ConsolidatedMetric(candid="test", survey="test")
        for v in STEP_MAPPER.values():
            sm = StepMetric(
                received=datetime.now(), sent=datetime.now(), execution_time=0.0
            )
            cm[v] = sm
        self.assertTrue(cm.is_bingo(PIPELINE_ORDER["ZTF"]))

    @mock.patch("redis.connection.ConnectionPool.get_connection")
    def test_is_bingo_false(self, mock_connection):
        cm = ConsolidatedMetric(candid="test", survey="test")
        fields = list(STEP_MAPPER.values())
        for v in fields[0:2]:
            sm = StepMetric(
                received=datetime.now(), sent=datetime.now(), execution_time=0.0
            )
            cm[v] = sm
        self.assertFalse(cm.is_bingo(PIPELINE_ORDER["ZTF"]))

    def test_get_prev_step(self):
        cm = ConsolidatedMetric(candid="test", survey="test")
        prev_step = cm._get_prev_step(PIPELINE_ORDER["ZTF"], "IngestionStep")
        self.assertEqual(prev_step, "SortingHatStep")

        prev_step = cm._get_prev_step(PIPELINE_ORDER["ZTF"], "SortingHatStep")
        self.assertEqual(prev_step, None)

    @mock.patch("redis.connection.ConnectionPool.get_connection")
    def test_compute_queue_time_between(self, mock_connection):
        step_1 = StepMetric(
            execution_time=1,
            received=datetime(2022, 1, 1, 12, 0, 0, 0),
            sent=datetime(2022, 1, 1, 13, 0, 0, 0),
        )

        step_2 = StepMetric(
            execution_time=1,
            received=datetime(2022, 1, 1, 14, 0, 0, 0),
            sent=datetime(2022, 1, 1, 15, 0, 0, 0),
        )

        cm = ConsolidatedMetric(
            candid="test",
            survey="test",
            s3=None,
            early_classifier=None,
            watchlist=None,
            xmatch=None,
            features=step_1,
            late_classifier=step_2,
        )

        queue_time = cm.compute_queue_time_between("FeaturesComputer", "LateClassifier")
        self.assertEqual(3600.0, queue_time)

        bad = cm.compute_queue_time_between("EarlyClassifier", "LateClassifier")
        self.assertEqual(None, bad)

    @mock.patch("redis.connection.ConnectionPool.get_connection")
    def test_compute_queue_time(self, mock_connection):
        cm = ConsolidatedMetric(candid="test", survey="test")

        cm.sorting_hat = StepMetric(
            execution_time=1,
            received=datetime(2022, 1, 1, 12, 0, 0, 0),
            sent=datetime(2022, 1, 1, 13, 0, 0, 0),
        )
        cm.ingestion = StepMetric(
            execution_time=1,
            received=datetime(2022, 1, 1, 14, 0, 0, 0),
            sent=datetime(2022, 1, 1, 15, 0, 0, 0),
        )

        queue_time = cm.compute_queue_time(PIPELINE_ORDER["ZTF"], "IngestionStep")
        self.assertEqual(3600.0, queue_time)

        status = cm.status(PIPELINE_ORDER["ZTF"])
        self.assertEqual("2/8", status)

    @mock.patch("redis.connection.ConnectionPool.get_connection")
    def test_compute_accumulated_time(self, mock_connection):
        cm = ConsolidatedMetric(candid="test", survey="test")

        cm.sorting_hat = StepMetric(
            execution_time=1,
            received=datetime(2022, 1, 1, 12, 0, 0, 0),
            sent=datetime(2022, 1, 1, 13, 0, 0, 0),
        )
        cm.ingestion = StepMetric(
            execution_time=1,
            received=datetime(2022, 1, 1, 14, 0, 0, 0),
            sent=datetime(2022, 1, 1, 15, 0, 0, 0),
        )

        cm.xmatch = StepMetric(
            execution_time=1,
            received=datetime(2022, 1, 1, 15, 0, 0, 0),
            sent=datetime(2022, 1, 1, 16, 0, 0, 0),
        )

        cm.features = StepMetric(
            execution_time=1,
            received=datetime(2022, 1, 1, 17, 0, 0, 0),
            sent=datetime(2022, 1, 1, 18, 0, 0, 0),
        )

        cm.late_classifier = StepMetric(
            execution_time=1,
            received=datetime(2022, 1, 1, 18, 0, 0, 0),
            sent=datetime(2022, 1, 1, 19, 0, 0, 0),
        )

        accum_time = cm.compute_accumulated_time(
            PIPELINE_ORDER["ZTF"], "LateClassifier"
        )
        self.assertEqual(7205.0, accum_time)

        status = cm.status(PIPELINE_ORDER["ZTF"])
        self.assertEqual("5/8", status)
