import pytest
import unittest

from xmatch_step.step import XmatchStep
from xmatch_step import XmatchStep, XmatchClient, Step, Object, Allwise, Xmatch
from schema import SCHEMA
from tests.data.messages import generate_input_batch


DB_CONFIG = {
    "SQL": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "PORT": 5432,
        "DB_NAME": "postgres",
    }
}

PRODUCER_CONFIG = {
    "TOPIC": "test",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": SCHEMA,
}

XMATCH_CONFIG = {
    "CATALOG": {
        "name": "allwise",
        "columns": [
            "AllWISE",
            "RAJ2000",
            "DEJ2000",
            "W1mag",
            "W2mag",
            "W3mag",
            "W4mag",
            "e_W1mag",
            "e_W2mag",
            "e_W3mag",
            "e_W4mag",
            "Jmag",
            "e_Jmag",
            "Hmag",
            "e_Hmag",
            "Kmag",
            "e_Kmag",
        ],
    }
}


@pytest.mark.usefixtures("psql_service")
@pytest.mark.usefixtures("kafka_service")
class StepXmatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # this step only for setup db
        cls.step_config = {
            "DB_CONFIG": DB_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "STEP_METADATA": {
                "STEP_VERSION": "xmatch",
                "STEP_ID": "xmatch",
                "STEP_NAME": "xmatch",
                "STEP_COMMENTS": "xmatch",
            },
            "XMATCH_CONFIG": XMATCH_CONFIG,
            "RETRIES": 3,
            "RETRY_INTERVAL": 1,
        }
        cls.step = XmatchStep(config=cls.step_config)

    @classmethod
    def tearDownClass(cls):
        cls.step.driver.drop_db()
        cls.step.driver.session.close()

    def setUp(self):
        self.step.driver.create_db()

    def tearDown(self):
        self.step.driver.session.close()
        self.step.driver.drop_db()

    def test_insert_step_metadata(self):
        self.step.insert_step_metadata()
        self.assertEqual(len(self.step.driver.query(Step).all()), 1)

    def test_execute(self):
        batch = generate_input_batch(20)  # I want 20 light  curves
        step = XmatchStep(config=self.step_config)
        step.execute(batch)


