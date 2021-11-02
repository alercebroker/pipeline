import pytest
import unittest
from apf.producers.kafka import KafkaProducer
from unittest import mock
from generic_save_step.step import GenericSaveStep


@pytest.mark.usefixtures("mongo_service")
class MongoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        db_config = {
            "MONGO": {
                "HOST": "localhost",
                "USER": "",
                "PASSWORD": "",
                "PORT": 27017,
                "DB_NAME": "test",
            }
        }
        self.step_config = {
            "DB_CONFIG": db_config,
            "STEP_METADATA": {
                "STEP_ID": "ingestion",
                "STEP_NAME": "ingestion",
                "STEP_VERSION": "test",
                "STEP_COMMENTS": "",
            },
        }
        self.mock_producer = mock.create_autospec(KafkaProducer)
        self.step = GenericSaveStep(
            config=self.step_config,
            producer=self.mock_producer,
        )

    @classmethod
    def tearDownClass(self):
        self.step.db.drop_db()

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
        self.step.driver.create_db()

    def tearDown(self):
        self.step.db.drop_db()
