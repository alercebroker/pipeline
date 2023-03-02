import unittest
import pytest
import pandas as pd
from unittest import mock

from apf.producers import KafkaProducer
from magstats_step.utils.multi_driver.connection import MultiDriverConnection
from magstats_step.step import MagstatsStep

from data.messages import (
    LC_MESSAGE,
)

DB_CONFIG = {
    "PSQL": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "PORT": 5432,
        "DB_NAME": "postgres",
    },
    "MONGO": {
        "HOST": "localhost",
        "USER": "test_user",
        "PASSWORD": "test_password",
        "PORT": 27017,
        "DATABASE": "test_db",
    },
}

class StepTestCase(unittest.TestCase):
    def setUp(self) -> None:
        step_config = {
            "DB_CONFIG": DB_CONFIG,
            "STEP_METADATA": {
                "STEP_ID": "magstats",
                "STEP_NAME": "magstats",
                "STEP_VERSION": "test",
                "STEP_COMMENTS": "test version",
            },
        }
        mock_database_connection = mock.create_autospec(MultiDriverConnection)
        mock_database_connection.connect.return_value = None
        mock_producer = mock.create_autospec(KafkaProducer)
        self.step = MagstatsStep(
            config=step_config,
            db_connection=mock_database_connection,
            producer=mock_producer,
        )

    def tearDown(self) -> None:
        del self.step

    def test_step(self):
        self.step.execute(LC_MESSAGE)
        # Verify magstats insert call
        return

