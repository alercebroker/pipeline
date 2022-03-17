import pytest
import unittest

from ingestion_step.utils.multi_driver.connection import MultiDriverConnection
import db_plugins.db.mongo.models as mongo_models
import db_plugins.db.sql.models as psql_models


CONFIG = {
    "PSQL": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "PORT": 5432,
        "DB_NAME": "postgres"
    },
    "MONGO": {
        "HOST": "localhost",
        "USER": "test_user",
        "PASSWORD": "test_password",
        "PORT": 27017,
        "DATABASE": "test_db",
    }
}


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("psql_service")
class MultiDriverTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = MultiDriverConnection(CONFIG)
        cls.driver.connect()
        cls.driver.create_db()

    @classmethod
    def tearDownClass(cls):
        cls.driver.drop_db()

    def test_get_or_create(self):
        ins, created = self.driver.query().get_or_create(
            model=mongo_models.Object,
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
        self.assertTrue(created)

    def test_psql_get_or_create(self):
        pass