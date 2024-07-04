import unittest
from unittest import mock
from apf.consumers import KafkaConsumer
import pandas as pd
from apf.producers import KafkaProducer
from sorting_hat_step.step import SortingHatStep
from .data.batch import generate_alerts_batch
from sorting_hat_step.database import MongoConnection
from pymongo.database import Database


class SortingHatStepTestCase(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "FEATURE_FLAGS": {
                "RUN_CONESEARCH": True,
                "USE_PSQL": False,
                "USE_MONGO": True,
            },
        }
        self.mock_db = mock.create_autospec(MongoConnection)
        self.mock_db.database = mock.create_autospec(Database)
        self.mock_producer = mock.create_autospec(KafkaProducer)
        self.mock_consumer = mock.create_autospec(KafkaConsumer)
        self.step = SortingHatStep(
            config=self.step_config,
            mongo_connection=self.mock_db,
        )

    def tearDown(self):
        del self.mock_db
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


class RunConesearchTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_db = mock.create_autospec(MongoConnection)
        self.mock_db.database = mock.create_autospec(Database)
        self.dataframe = pd.DataFrame(
            [[1, 2, 3, 4, 5]], columns=["ra", "dec", "oid", "tid", "aid"]
        )

    @mock.patch("sorting_hat_step.step.wizard")
    def test_run_conesearch_explicit_True(self, mock_wizzard):
        step_config = {
            "PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "FEATURE_FLAGS": {
                "RUN_CONESEARCH": True,
                "USE_PSQL": False,
                "USE_MONGO": True,
            },
        }

        step = SortingHatStep(config=step_config, mongo_connection=self.mock_db)
        step.add_aid(self.dataframe)  # the wizzard is mocked

        mock_wizzard.find_existing_id.assert_called_once()
        mock_wizzard.find_id_by_conesearch.assert_called_once()
        mock_wizzard.generate_new_id.assert_called_once()

    @mock.patch("sorting_hat_step.step.wizard")
    def test_run_conesearch_default_True(self, mock_wizzard):
        step_config = {
            "PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "FEATURE_FLAGS": {
                "RUN_CONESEARCH": True,
                "USE_PSQL": False,
                "USE_MONGO": True,
            },
        }

        step = SortingHatStep(config=step_config, mongo_connection=self.mock_db)
        step.add_aid(self.dataframe)  # the wizzard is mocked

        mock_wizzard.find_existing_id.assert_called_once()
        mock_wizzard.find_id_by_conesearch.assert_called_once()
        mock_wizzard.generate_new_id.assert_called_once()

    @mock.patch("sorting_hat_step.step.wizard")
    def test_dont_run_conesearch(self, mock_wizzard):
        step_config = {
            "PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "FEATURE_FLAGS": {
                "RUN_CONESEARCH": False,
                "USE_PSQL": False,
                "USE_MONGO": True,
            },
        }
        step = SortingHatStep(config=step_config, mongo_connection=self.mock_db)
        step.add_aid(self.dataframe)  # the wizzard is mocked

        mock_wizzard.find_existing_id.assert_called_once()
        mock_wizzard.find_id_by_conesearch.assert_not_called()
        mock_wizzard.generate_new_id.assert_called_once()

    @mock.patch("sorting_hat_step.step.wizard")
    def test_run_post_execute(self, mock_wizzard):
        step_config = {
            "PRODUCER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock"},
            "FEATURE_FLAGS": {
                "RUN_CONESEARCH": False,
                "USE_PSQL": False,
                "USE_MONGO": True,
            },
        }
        step = SortingHatStep(config=step_config, mongo_connection=self.mock_db)
        step.post_execute(self.dataframe)  # the wizzard is mocked

        mock_wizzard.insert_empty_objects.assert_called_once()
