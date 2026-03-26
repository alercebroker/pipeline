import os
import unittest
import logging

import numpy as np
import pytest

from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.sql.models import Probability
from sqlalchemy import select

from rubin_stamp_classifier_step.step import StampClassifierStep
from rubin_stamp_classifier_step.db.db import class_id_to_name
from .data.load_msgs_sample import load_sample_messages


psql_config_direct = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}

psql_config_pgbouncer = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5433,
    "DB_NAME": "postgres",
    "POOLCLASS": "NullPool",
}

step_config = {
    "LOGGING_LEVEL": "DEBUG",
    "DB_CONFIG": {
        "USER": psql_config_pgbouncer["USER"],
        "PASSWORD": psql_config_pgbouncer["PASSWORD"],
        "HOST": psql_config_pgbouncer["HOST"],
        "PORT": psql_config_pgbouncer["PORT"],
        "DB_NAME": psql_config_pgbouncer["DB_NAME"],
        "SCHEMA": "public",
        "POOLCLASS": "NullPool",
    },
    "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},
    "PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"},
    "MODEL_VERSION": "",
    "MODEL_CONFIG": {
        "MODEL_PATH": os.environ["TEST_RUBIN_STAMP_CLASSIFIER_STEP_MODEL_PATH"],
        "CLS_ID": 1,
    },
}


@pytest.mark.usefixtures("pgbouncer_service")
class TestRubinStampClassifierStep(unittest.TestCase):
    """Test class for Rubin Stamp Classifier Step."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all test methods."""
        cls.db_sql = PsqlDatabase(psql_config_pgbouncer)

    def setUp(self):
        # DDL via direct connection.
        self.db_sql_direct = PsqlDatabase(psql_config_direct)
        self.db_sql_direct.create_db()

        # Queries and step use pgbouncer.
        # create_db() seeds taxonomy via _initial_data.py (CLS_ID=1 = rubin).
        self.db_sql = PsqlDatabase(psql_config_pgbouncer)
        self.step = StampClassifierStep(
            config=step_config,
            level=step_config["LOGGING_LEVEL"],
        )

        self.sample_messages = load_sample_messages()
        self.N_CLASSES = len(self.step.dict_mapping_classes)

    def tearDown(self):
        """Clean up after each test."""
        self.db_sql_direct.drop_db()

    @classmethod
    def _set_logger(cls, settings):
        """Set up the logger with the appropriate level and format."""
        level = logging.INFO
        if settings.get("LOGGING_DEBUG"):
            level = logging.DEBUG

        logger = logging.getLogger("alerce")
        logger.setLevel(level)

        fmt = logging.Formatter(
            "%(asctime)s %(levelname)7s %(name)36s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(level)

        logger.addHandler(handler)
        return logger

    def test_pre_execute(self):
        """Test the pre_execute method with valid messages."""
        processed_messages = self.step.pre_execute(self.sample_messages)

        # Check if the processed messages are in the expected format
        self.assertIsInstance(processed_messages, list)
        #self.assertEqual(len(processed_messages), len(self.sample_messages))

        required_fields = [
            "diaObjectId",
            "diaSourceId",
            "ssObjectId",
            "midpointMjdTai",
            "ra",
            "dec",
            "airmass",
            "magLim",
            "psfFlux",
            "psfFluxErr",
            "scienceFlux",
            "scienceFluxErr",
            "seeing",
            "snr",
            "visit_image",
            "difference_image",
            "reference_image",
        ]
        for message in processed_messages:
            # Check if each message has the required fields
            for field in required_fields:
                self.assertIn(field, message)

            # Check if the visit_image, difference_image, and reference_image are numpy arrays
            self.assertIsInstance(message["visit_image"], np.ndarray)
            self.assertIsInstance(message["difference_image"], np.ndarray)
            self.assertIsInstance(message["reference_image"], np.ndarray)

    def test_execute(self):
        """Test the execute method with valid messages."""
        processed_messages = self.step.pre_execute(self.sample_messages)
        results = self.step.execute(processed_messages)

        # Check if the results are in the expected format
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(processed_messages))

        for result in results:
            self.assertIn("probabilities", result)

            # Check that probabilities sum to 1
            probabilities = result["probabilities"].values()
            self.assertAlmostEqual(sum(probabilities), 1.0, places=5)

    def test_post_execute(self):
        """Test the post_execute method with valid messages."""
        processed_messages = self.step.pre_execute(self.sample_messages)
        execute_result = self.step.execute(processed_messages)
        self.step.post_execute(execute_result)

        # Check if the results are stored in the database
        for result in execute_result:
            oid = result["diaObjectId"] if result["diaObjectId"] != 0 else result["ssObjectId"]
            with self.db_sql.session() as session:
                query = select(Probability).where(Probability.oid == oid)
                db_result = session.execute(query).scalars().all()
            self.assertIsNotNone(db_result)
            self.assertEqual(len(db_result), self.N_CLASSES)

            # Check if the stored probabilities match the result
            for probability in db_result:
                self.assertIn(
                    class_id_to_name(probability.class_id, self.step.class_taxonomy), result["probabilities"]
                )
                self.assertAlmostEqual(
                    probability.probability,
                    result["probabilities"][class_id_to_name(probability.class_id, self.step.class_taxonomy)],
                    places=5,
                )
