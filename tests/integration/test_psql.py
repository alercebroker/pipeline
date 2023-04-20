import unittest
import numpy as np
import pickle
import pytest

from unittest import mock
from lc_classification.step import LateClassifier, KafkaProducer


@pytest.mark.usefixtures("psql_service")
class PSQLIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.step_config = {
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
        pass

    def setUp(self):
        self.probabilities = {}
        with open("tests/unittest/response_batch.pickle", "rb") as f:
            self.probabilities = pickle.load(f)


    def tearDown(self):
        pass

