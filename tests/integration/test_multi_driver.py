import pytest
import unittest
import db_plugins.db.mongo.models as mongo_models
import db_plugins.db.sql.models as psql_models

from ingestion_step.utils.multi_driver.connection import MultiDriverConnection
from pymongo.cursor import Cursor as MongoCursor

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
        cls.n_objects_mongo = 100
        cls.insert_data_mongo()

    @classmethod
    def insert_data_mongo(cls):
        for o in range(0, cls.n_objects_mongo):
            cls.driver.mongo_driver.query().get_or_create(
                model=mongo_models.Object,
                filter_by={
                    "aid": f"alerce{o}",
                    "oid": "ZTF1",
                    "lastmjd": 1,
                    "firstmjd": 1,
                    "ndet": 1,
                    "meanra": 0,
                    "meandec": 0,
                },
                _id=f"alerce{o}",
            )

    @classmethod
    def tearDownClass(cls):
        cls.driver.drop_db()

    def test_find_all(self):
        mongo_objects = self.driver.query(mongo_models.Object, engine="mongo").find_all(paginate=False)
        self.assertIsInstance(mongo_objects, MongoCursor)
        mongo_objects = [x for x in mongo_objects]
        self.assertEqual(len(mongo_objects), self.n_objects_mongo)

