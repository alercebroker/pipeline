import pytest
import unittest
import db_plugins.db.mongo.models as mongo_models
import db_plugins.db.sql.models as psql_models

from ingestion_step.utils.multi_driver.connection import MultiDriverConnection

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
        cls.n_objects_psql = 100
        cls.insert_data_mongo()
        cls.insert_data_psql()

    @classmethod
    def insert_data_mongo(cls):
        for o in range(0, cls.n_objects_mongo):
            cls.driver.mongo_driver.query().get_or_create(
                model=mongo_models.Object,
                filter_by={
                    "aid": f"alerce{o}",
                    "oid": f"ZTF{o}",
                    "lastmjd": 1,
                    "firstmjd": 1,
                    "ndet": 1,
                    "meanra": 0,
                    "meandec": 0,
                },
                _id=f"alerce{o}",
            )

    @classmethod
    def insert_data_psql(cls):
        for o in range(0, cls.n_objects_psql):
            cls.driver.psql_driver.query().get_or_create(
                psql_models.Object,
                {
                    "oid": f"ZTF{o}",
                    "lastmjd": 1,
                    "firstmjd": 1,
                    "ndet": 1,
                    "meanra": 0,
                    "meandec": 0,
                }
            )

    def test_bulk_insert_mongo(self):
        data = [
            {
                "aid": f"alerce{o}",
                "oid": f"ZTF{o}",
                "lastmjd": 1,
                "firstmjd": 1,
                "ndet": 1,
                "meanra": 0,
                "meandec": 0,
                "_id": f"alerce{o}"
            }
            for o in range(self.n_objects_mongo, self.n_objects_mongo+100)
        ]
        self.driver.query("Object", engine="mongo").bulk_insert(data)

    def test_bulk_insert_psql(self):
        data = [
            {
                "aid": f"alerce{o}",
                "oid": f"ZTF{o}",
                "lastmjd": 1,
                "firstmjd": 1,
                "ndet": 1,
                "meanra": 0,
                "meandec": 0,
            }
            for o in range(self.n_objects_mongo, self.n_objects_psql+100)
        ]
        self.driver.query("Object", engine="psql").bulk_insert(data)

    def test_bulk_update_mongo(self):
        data = [
            {
                "aid": f"alerce{o}",
                "oid": f"ZTF{o}",
                "lastmjd": 10,
                "firstmjd": 12,
                "ndet": 10,
                "meanra": 0,
                "meandec": 0,
            }
            for o in range(0, 10)
        ]
        filter_by = [{"_id": f"alerce{x}"} for x in range(0, 10)]
        self.driver.query("Object", engine="mongo").bulk_update(data, filter_by)

    def test_bulk_update_psql(self):
        data = [
            {
                "oid": f"ZTF{o}",
                "lastmjd": 10,
                "firstmjd": 12,
                "ndet": 10,
                "meanra": 0,
                "meandec": 0,
            }
            for o in range(0, 10)
        ]
        filter_by = [{"_id": f"ZTF{x}"} for x in range(0, 10)]
        self.driver.query("Object", engine="psql").bulk_update(data, filter_by)

    def test_find_all_mongo(self):
        mongo_objects = self.driver.query("Object", engine="mongo").find_all(paginate=False)
        self.assertIsInstance(mongo_objects, list)
        mongo_objects = [x for x in mongo_objects]
        self.assertEqual(len(mongo_objects), 200)

        filter_by = {"aid": {"$in": ["alerce1", "alerce2"]}}
        mongo_objects = self.driver.query("Object", engine="mongo").find_all(filter_by=filter_by, paginate=False)
        self.assertIsInstance(mongo_objects, list)
        mongo_objects = [x for x in mongo_objects]
        self.assertEqual(len(mongo_objects), 2)

    def test_find_all_psql(self):
        psql_objects = self.driver.query("Object", engine="psql").find_all(paginate=False)
        self.assertIsInstance(psql_objects, list)
        self.assertEqual(len(psql_objects), 200)

        filter_by = {"aid": {"$in": ["ZTF0", "ZTF3"]}}
        psql_objects = self.driver.query("Object", engine="psql").find_all(filter_by=filter_by, paginate=False)
        self.assertIsInstance(psql_objects, list)
        self.assertEqual(len(psql_objects), 2)

