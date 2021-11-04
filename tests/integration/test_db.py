import pytest
import unittest
from apf.producers.kafka import KafkaProducer
from unittest import mock
from generic_save_step.step import GenericSaveStep
from db_plugins.db.mongo.models import Object, Detection


@pytest.mark.usefixtures("mongo_service")
class MongoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        db_config = {
            "HOST": "localhost",
            "USER": "testo",
            "PASSWORD": "passu",
            "PORT": 27017,
            "DATABASE": "test_db",
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
        self.step.driver.drop_db()

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
        self.insert_objects()
        self.insert_detections()

    def tearDown(self):
        self.step.driver.drop_db()

    def insert_objects(self):
        ins, created = self.step.driver.query().get_or_create(
            model=Object,
            filter_by={
                "aid": "alerce1",
                "oid": "ZTF1",
                "lastmjd": 1,
                "firstmjd": 1,
                "ndet": 1,
                "meanra": 0,
                "meandec": 0,
            },
            _id="alerce1",
        )

    def insert_detections(self):
        ins, created = self.step.driver.query().get_or_create(
            model=Detection,
            filter_by={
                "aid": "alerce1",
                "tid": "ZTF",
                "candid": "candid1",
                "mjd": 1,
                "fid": 1,
                "ra": 0,
                "dec": 0,
                "rb": 0,
                "mag": 0,
                "e_mag": 0,
                "rfid": 0,
                "e_ra": 0,
                "e_dec": 0,
                "isdiffpos": 0,
                "corrected": 0,
                "parent_candid": 0,
                "has_stamp": 0,
                "step_id_corr": 0,
                "rbversion": 0,
            },
        )

    def test_get_objects(self):
        objs = self.step.get_objects(["alerce1"])
        self.assertEqual(len(objs), 1)

    def test_get_detections(self):
        dets = self.step.get_detections(["alerce1"])
        self.assertEqual(len(dets), 1)
