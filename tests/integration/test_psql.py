import unittest
from unittest import mock

import pytest
from s3_step.step import S3Step, Step
from apf.consumers import GenericConsumer


@pytest.mark.usefixtures("psql_service")
class PSQLIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        cls.step_config = {
            "DB_CONFIG": db_config,
            "STEP_METADATA": {
                "STEP_ID": "",
                "STEP_NAME": "",
                "STEP_VERSION": "",
                "STEP_COMMENTS": "",
            },
            "API_URL": "",
            "N_RETRY": 5,
        }
        mock_consumer = mock.create_autospec(GenericConsumer)
        mock_message = mock.MagicMock()
        mock_message.value.return_value = b"fake"
        mock_consumer.messages = [mock_message]
        cls.step = S3Step(
            config=cls.step_config,
            test_mode=True,
            consumer=mock_consumer,
        )

    @classmethod
    def tearDownClass(cls):
        cls.step.db.drop_db()
        cls.step.db.session.close()

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
        }
        self.step.db.create_db()

    def tearDown(self):
        self.step.db.session.close()
        self.step.db.drop_db()

    def test_insert_step_metadata(self):
        self.step.insert_step_metadata()
        self.assertEqual(len(self.step.db.query(Step).all()), 1)
