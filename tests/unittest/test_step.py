import unittest
from unittest import mock
from apf.consumers import KafkaConsumer
from apf.core.step import logging

import pandas as pd
from apf.producers import KafkaProducer
from db_plugins.db.mongo.connection import MongoConnection
from sorting_hat_step import SortingHatStep
from data.batch import generate_alerts_batch


class SortingHatStepTestCase(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "DB_CONFIG": {},
            "PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "STEP_METADATA": {
                "STEP_ID": "",
                "STEP_NAME": "",
                "STEP_VERSION": "",
                "STEP_COMMENTS": "",
            },
        }
        self.mock_database_connection = mock.create_autospec(MongoConnection)
        self.mock_producer = mock.create_autospec(KafkaProducer)
        self.mock_consumer = mock.create_autospec(KafkaConsumer)
        self.step = SortingHatStep(
            config=self.step_config,
            db_connection=self.mock_database_connection,
            level=logging.DEBUG,
        )

    def tearDown(self):
        del self.mock_database_connection
        del self.mock_producer
        del self.step

    @mock.patch("sorting_hat_step.step.SortingHatStep._add_metrics")
    @mock.patch("sorting_hat_step.utils.wizard.oid_query")
    def test_execute(self, mock_query, _):
        alerts = generate_alerts_batch(100)
        mock_query.return_value = "aid"
        result = self.step.execute(alerts)
        assert "aid" in result

    @mock.patch("sorting_hat_step.utils.wizard.oid_query")
    def test_pre_produce(self, mock_query):
        mock_query.return_value = "aid"
        self.step.producer = mock.MagicMock(KafkaProducer)
        alerts = generate_alerts_batch(100)
        parsed = self.step.parser.parse(alerts)
        parsed = pd.DataFrame(parsed)
        alerts = self.step.add_aid(parsed)
        result = self.step.pre_produce(alerts)
        assert len(result) == len(alerts)
        for msg in result:
            assert isinstance(msg, dict)

    def test_add_metrics(self):
        dataframe = pd.DataFrame(
            [[1, 2, 3, 4, 5]], columns=["ra", "dec", "oid", "tid", "aid"]
        )
        self.step._add_metrics(dataframe)
        assert self.step.metrics["ra"] == [1]
        assert self.step.metrics["dec"] == [2]
        assert self.step.metrics["oid"] == [3]
        assert self.step.metrics["tid"] == [4]
        assert self.step.metrics["aid"] == [5]
