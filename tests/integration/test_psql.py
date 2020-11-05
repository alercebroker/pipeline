import unittest
import numpy as np
import pickle


from unittest import mock
from db_plugins.db.sql.models import Step, Object, Probability
from lc_classification.step import LateClassifier, KafkaProducer


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
                "CLASSIFIER_NAME": "hrf_test",
            },
            "PRODUCER_CONFIG": {},
            "N_RETRY": 5,
        }
        self.mock_producer = mock.create_autospec(KafkaProducer)
        self.step = LateClassifier(
            config=self.step_config,
            producer=self.mock_producer,
            test_mode=True
        )

    @classmethod
    def tearDownClass(self):
        self.step.driver.drop_db()
        self.step.driver.session.close()

    def setUp(self):
        self.probabilities = {}
        with open("tests/unittest/response_batch.pickle", "rb") as f:
            self.probabilities = pickle.load(f)

        self.step.driver.create_db()

    def tearDown(self):
        self.step.driver.session.close()
        self.step.driver.drop_db()

    def test_insert_step_metadata(self):
        self.step.insert_step_metadata()
        self.assertEqual(len(self.step.driver.query(Step).all()), 1)

    def test_insert_db(self):
        db_results = self.step.process_results(self.probabilities)
        oids = [f"ZTF{i}" for i in range(10)]

        for i in oids:
            self.step.driver.session.query().get_or_create(
                Object, {
                    "oid": i
                }
            )
        db_results["oid"] = db_results["oid"].map(lambda x: f"ZTF{x}")

        self.step.insert_db(db_results, oids)
        self.assertEqual(len(self.step.driver.query(Object).all()), 10)
        self.assertEqual(len(self.step.driver.query(Probability).all()), 330)
