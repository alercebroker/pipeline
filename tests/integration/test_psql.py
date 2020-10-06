import unittest
from unittest import mock
import subprocess
import time
from features.step import FeaturesComputer, KafkaProducer, Step, Feature, FeatureVersion
from db_plugins.db.sql.models import Object


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
                "STEP_ID": "features",
                "STEP_NAME": "features",
                "STEP_VERSION": "test",
                "STEP_COMMENTS": "",
                "FEATURE_VERSION": "feature",
            },
        }
        self.mock_producer = mock.create_autospec(KafkaProducer)
        self.mock_custom_hierarchical_extractor = mock.create_autospec(FeaturesComputer)
        self.step = FeaturesComputer(
            config=self.step_config,
            features_computer=self.mock_custom_hierarchical_extractor,
            producer=self.mock_producer,
            test_mode=True,
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
        oid = "oid"
        features = {"test_1": 0}
        preprocess_id = "pre"
        obj = Object(oid=oid)
        self.step.db.session.add(obj)
        self.step.db.session.commit()
        self.step.insert_step_metadata()
        self.step.config["STEP_METADATA"] = {
            "STEP_ID": preprocess_id,
            "STEP_NAME": "preprocess",
            "STEP_VERSION": "test",
            "STEP_COMMENTS": "",
            "FEATURE_VERSION": "feature",
        }
        self.step.insert_step_metadata()
        self.step.insert_db(oid, features, preprocess_id)
        self.assertEqual(len(self.step.db.query(Feature).all()), 1)
        self.assertEqual(len(self.step.db.query(FeatureVersion).all()), 1)
