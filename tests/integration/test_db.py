import pytest
import unittest
import pandas as pd

from apf.producers.kafka import KafkaProducer
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from ingestion_step.step import IngestionStep
from unittest import mock


DB_CONFIG = {
    "HOST": "localhost",
    "USER": "test_user",
    "PASSWORD": "test_password",
    "PORT": 27017,
    "DATABASE": "test_db",
}


@pytest.mark.usefixtures("mongo_service")
class MongoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step_config = {
            "DB_CONFIG": DB_CONFIG,
            "STEP_METADATA": {
                "STEP_ID": "ingestion",
                "STEP_NAME": "ingestion",
                "STEP_VERSION": "test",
                "STEP_COMMENTS": "",
            },
        }
        cls.mock_producer = mock.create_autospec(KafkaProducer)
        cls.step = IngestionStep(
            config=cls.step_config,
            producer=cls.mock_producer,
        )

    @classmethod
    def tearDownClass(cls):
        cls.step.driver.drop_db()

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
        self.insert_non_detections()

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

    def insert_non_detections(self):
        ins, created = self.step.driver.query().get_or_create(
            model=NonDetection,
            filter_by={
                "aid": "alerce1",
                "tid": "ZTF",
            },
            mjd=1,
            diffmaglim=1,
            fid=1,
        )

    def test_get_objects(self):
        objs = self.step.get_objects(["alerce1"])
        self.assertEqual(len(objs), 1)

    def test_get_detections(self):
        dets = self.step.get_detections(["alerce1"])
        self.assertEqual(len(dets), 1)

    def test_get_non_detections(self):
        non_dets = self.step.get_detections(["alerce1"])
        self.assertEqual(len(non_dets), 1)

    def test_insert_objects(self):
        objs = pd.DataFrame(
            {
                "aid": ["alerce1", "alerce2"],
                "oid": ["ZTF2", "ATLAS1"],
                "lastmjd": [1, 2],
                "firstmjd": [1, 2],
                "ndet": [1, 2],
                "meanra": [1, 2],
                "meandec": [1, 2],
                "new": [False, True],
            }
        )
        self.step.insert_objects(objs)
        found = list(
            self.step.driver.query().find_all(
                model=Object, filter_by={"aid": "alerce1"}, paginate=False
            )
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["oid"], objs.iloc[0].oid)
        found = list(
            self.step.driver.query().find_all(
                model=Object, filter_by={"aid": "alerce2"}, paginate=False
            )
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["oid"], objs.iloc[1].oid)

    def test_insert_detections(self):
        objs = pd.DataFrame(
            {
                "aid": ["alerce2"],
                "tid": ["ZTF"],
                "candid": ["candid1"],
                "mjd": [1],
                "fid": [1],
                "ra": [0],
                "dec": [0],
                "rb": [0],
                "mag": [0],
                "e_mag": [0],
                "rfid": [0],
                "e_ra": [0],
                "e_dec": [0],
                "isdiffpos": [0],
                "corrected": [0],
                "parent_candid": [0],
                "has_stamp": [0],
                "step_id_corr": [0],
                "rbversion": [0],
            }
        )
        self.step.insert_detections(objs)
        found = list(
            self.step.driver.query().find_all(
                model=Detection, filter_by={"aid": "alerce2"}, paginate=False
            )
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["aid"], objs.iloc[0].aid)

    def test_insert_non_detections(self):
        objs = pd.DataFrame(
            {
                "aid": ["alerce2"],
                "tid": ["ZTF"],
                "mjd": [1],
                "fid": [1],
                "diffmaglim": [0],
            }
        )
        self.step.insert_non_detections(objs)
        found = list(
            self.step.driver.query().find_all(
                model=NonDetection,
                filter_by={"aid": "alerce2"},
                paginate=False,
            )
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["aid"], objs.iloc[0].aid)
