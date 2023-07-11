import unittest
import pytest
from unittest import mock
import subprocess
import time
from earlyclassifier.step import (
    EarlyClassifier,
    datetime,
    requests,
    SQLConnection,
    Probability,
    Object,
    Step,
)
import earlyclassifier
import os
import random

FILE_PATH = os.path.dirname(__file__)


@pytest.mark.usefixtures("config_database")
class PSQLIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        db_config = {
            "SQL": {
                "ENGINE": "postgresql",
                "HOST": "localhost",
                "USER": "postgres",
                "PASSWORD": "postgres",
                "PORT": 5432,
                "DB_NAME": "postgres",
            }
        }
        self.step_config = {
            "DB_CONFIG": db_config,
            "STEP_METADATA": {
                "STEP_ID": "",
                "STEP_NAME": "",
                "STEP_VERSION": "",
                "STEP_COMMENTS": "",
                "CLASSIFIER_VERSION": "test",
                "CLASSIFIER_NAME": "stamp_test",
            },
            "API_URL": "",
            "N_RETRY": 5,
        }
        self.mock_session = mock.create_autospec(earlyclassifier.step.requests.Session)
        self.step = EarlyClassifier(
            config=self.step_config, request_session=self.mock_session, test_mode=True
        )

    @classmethod
    def tearDownClass(self):
        self.step.db.drop_db()
        self.step.db.session.close()

    def setUp(self):
        self.message = {
            "objectId": "ZTF1",
            "candidate": {
                "ndethist": 0,
                "ncovhist": 0,
                "jdstarthist": 2400000.5,
                "jdendhist": 2400000.5,
                "jd": 2400000.5,
                "ra": 0,
                "dec": 0,
            },
            "cutoutTemplate": {"stampData": b""},
            "cutoutScience": {"stampData": b""},
            "cutoutDifference": {"stampData": b""},
        }
        self.step.db.create_db()

    def tearDown(self):
        self.step.db.session.close()
        self.step.db.drop_db()

    def test_insert_step_metadata(self):
        self.step.insert_step_metadata()
        self.assertEqual(len(self.step.db.query(Step).all()), 1)

    def test_insert_db(self):
        probabilities = {
            "AGN": 1,
            "SN": 2,
            "bogus": 3,
            "asteroid": 4,
            "VS": 5,
        }
        object_data = self.step.get_default_object_values(self.message)
        self.step.insert_db(probabilities, self.message["objectId"], object_data)
        self.assertEqual(len(self.step.db.query(Object).all()), 1)
        self.assertEqual(len(self.step.db.query(Probability).all()), 5)
