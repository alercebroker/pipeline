import unittest
from unittest import mock
import subprocess
import docker
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
from settings import DB_CONFIG

FILE_PATH = os.path.dirname(__file__)


class PSQLIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.client = docker.from_env()
        self.container = self.client.containers.run(
            image="postgres",
            name="test",
            environment=[
                "POSTGRES_USER=postgres",
                "POSTGRES_PASSWORD=password",
                "POSTGRES_DB=test",
            ],
            detach=True,
            ports={"5432/tcp": 5432},
        )
        time.sleep(5)
        subprocess.run(
            [f'dbp initdb --settings_path {os.path.join(FILE_PATH, "settings.py")}'],
            shell=True,
        )
        self.step_config = {
            "DB_CONFIG": DB_CONFIG,
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
            config=self.step_config,
            request_session=self.mock_session,
        )
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

    def tearDown(self):
        self.step.db.session.close()
        time.sleep(5)
        self.container.stop()
        self.container.remove()

    def test_insert_step_metadata(self):
        self.assertEqual(len( self.step.db.query(Step).all() ), 1)

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
