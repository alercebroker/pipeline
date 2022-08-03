from consolidated_metrics_step.utils.metric import ConsolidatedMetric
from consolidated_metrics_step.utils.metric import STEP_MAPPER
from consolidated_metrics_step.utils.metric import StepMetric
from datetime import datetime
from unittest import mock

import unittest

PIPELINE_ORDER = {
    "ATLAS": {"S3Step": None, "SortingHatStep": {"IngestionStep": None}},
    "ZTF": {
        "EarlyClassifier": None,
        "S3Step": None,
        "WatchlistStep": None,
        "SortingHatStep": {
            "IngestionStep": {
                "XmatchStep": {"FeaturesComputer": {"LateClassifier": None}}
            }
        },
    },
}


class ConsolidatedMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline_1 = {"SortingHatStep": {"IngestionStep": None}}
        self.pipeline_2 = {
            "SortingHatStep": {
                "IngestionStep": None,
                "XmatchStep": None,
            }
        }

    @mock.patch("redis.connection.ConnectionPool.get_connection")
    def test_create_metric_model(self, mock_connection):
        cm = ConsolidatedMetric(candid="test", survey="test")
        assert cm.candid == "test"
        assert cm.survey == "test"

    # @mock.patch("redis.connection.ConnectionPool.get_connection")
    # def test_is_bingo_true(self, mock_connection):
    #     cm = ConsolidatedMetric(candid="test", survey="test")
    #     for v in STEP_MAPPER.values():
    #         sm = StepMetric(
    #             received=datetime.now(), sent=datetime.now(), execution_time=0.0
    #         )
    #         cm[v] = sm
    #     self.assertTrue(cm.is_bingo(PIPELINE_ORDER["ZTF"]))
    #
    # @mock.patch("redis.connection.ConnectionPool.get_connection")
    # def test_is_bingo_false(self, mock_connection):
    #     cm = ConsolidatedMetric(candid="test", survey="test")
    #     fields = list(STEP_MAPPER.values())
    #     for v in fields[0:2]:
    #         sm = StepMetric(
    #             received=datetime.now(), sent=datetime.now(), execution_time=0.0
    #         )
    #         cm[v] = sm
    #     self.assertFalse(cm.is_bingo(PIPELINE_ORDER["ZTF"]))
    #
    # @mock.patch("redis.connection.ConnectionPool.get_connection")
    # def test_compute_queue_time(self, mock_connection):
    #     dummy = StepMetric(
    #         execution_time=1,
    #         received=datetime(2022, 1, 1, 12, 0, 0, 0),
    #         sent=datetime(2022, 1, 1, 13, 0, 0, 0),
    #     )
    #     cm = ConsolidatedMetric(
    #         candid="test",
    #         survey="test",
    #         s3=dummy,
    #         early_classifier=dummy,
    #         watchlist=dummy,
    #         xmatch=dummy,
    #         features=dummy,
    #         late_classifier=dummy,
    #     )
    #
    #     cm.sorting_hat = StepMetric(
    #         execution_time=1,
    #         received=datetime(2022, 1, 1, 12, 0, 0, 0),
    #         sent=datetime(2022, 1, 1, 13, 0, 0, 0),
    #     )
    #     cm.ingestion = StepMetric(
    #         execution_time=1,
    #         received=datetime(2022, 1, 1, 14, 0, 0, 0),
    #         sent=datetime(2022, 1, 1, 15, 0, 0, 0),
    #     )
    #     self.assertDictEqual(
    #         cm.compute_queue_times(self.pipeline_1),
    #         {"SortingHatStep_IngestionStep": 3600.0},
    #     )
    #
    # @mock.patch("redis.connection.ConnectionPool.get_connection")
    # def test_bad_compute_queue_times(self, mock_connection):
    #     cm = ConsolidatedMetric(candid="test", survey="test")
    #     with self.assertRaises(Exception) as context:
    #         cm.compute_queue_times(PIPELINE_ORDER["ZTF"])
    #     self.assertIsInstance(context.exception, Exception)
    #
    # @mock.patch("redis.connection.ConnectionPool.get_connection")
    # def test_compute_total_time(self, mock_connection):
    #     cm = ConsolidatedMetric(candid="test", survey="test")
    #     cm.sorting_hat = StepMetric(
    #         execution_time=1,
    #         received=datetime(2022, 1, 1, 12, 0, 0, 0),
    #         sent=datetime(2022, 1, 1, 13, 0, 0, 0),
    #     )
    #     cm.ingestion = StepMetric(
    #         execution_time=1,
    #         received=datetime(2022, 1, 1, 14, 0, 0, 0),
    #         sent=datetime(2022, 1, 1, 15, 0, 0, 0),
    #     )
    #     self.assertEqual(cm.compute_total_time("sorting_hat", "ingestion"), 10800.0)
    #
    #     cm.ingestion = StepMetric(
    #         execution_time=1,
    #         received=datetime(2022, 1, 1, 14, 0, 0, 0),
    #         sent=datetime(2022, 1, 1, 12, 30, 0, 0),
    #     )
    #     self.assertEqual(cm.compute_total_time("sorting_hat", "ingestion"), 1800.0)
    #
    # @mock.patch("redis.connection.ConnectionPool.get_connection")
    # def test_bad_compute_total_time(self, mock_connection):
    #     cm = ConsolidatedMetric(candid="test", survey="test")
    #     with self.assertRaises(Exception) as context:
    #         cm.compute_total_time("sorting_hat", "ingestion")
    #     self.assertIsInstance(context.exception, Exception)
