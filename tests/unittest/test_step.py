import unittest
from unittest import mock
from unittest.mock import MagicMock
from apf.consumers import KafkaConsumer

import pandas as pd
from apf.producers import KafkaProducer
from db_plugins.db.mongo.connection import MongoConnection
from sorting_hat_step import SortingHatStep
from data.batch import generate_alerts_batch


class SortingHatStepTestCase(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "DB_CONFIG": {},
            "PRODUCER_CONFIG": {"fake": "fake"},
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
            consumer=self.mock_consumer,
            config=self.step_config,
            producer=self.mock_producer,
            db_connection=self.mock_database_connection,
        )

    def tearDown(self):
        del self.mock_database_connection
        del self.mock_producer
        del self.step

    @mock.patch("sorting_hat_step.step.SortingHatStep.produce")
    def test_execute(self, mock_produce: MagicMock):
        alerts = generate_alerts_batch(100)
        self.step.execute(alerts)
        mock_produce.assert_called()

    def test_produce(self):
        alerts = generate_alerts_batch(100)
        parsed = self.step.parser.parse(alerts)
        parsed = pd.DataFrame(parsed)
        alerts = self.step.add_aid(parsed)
        self.step.produce(alerts)
        self.step.producer.produce.assert_called()
        self.assertEqual(self.step.producer.produce.call_count, len(alerts))
