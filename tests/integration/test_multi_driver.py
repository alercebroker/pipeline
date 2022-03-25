import pytest
import unittest

from ingestion_step.utils.multi_driver.connection import MultiDriverConnection
from tests.unittest.data.messages import (
    generate_random_detections,
    generate_random_objects,
)

CONFIG = {
    "PSQL": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "PORT": 5432,
        "DB_NAME": "postgres",
    },
    "MONGO": {
        "HOST": "localhost",
        "USER": "test_user",
        "PASSWORD": "test_password",
        "PORT": 27017,
        "DATABASE": "test_db",
    },
}


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("psql_service")
class MultiDriverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = MultiDriverConnection(CONFIG)
        cls.driver.connect()
        cls.driver.create_db()
        cls.n_insert_objects_mongo = 100
        cls.n_insert_detections_mongo = 100
        cls.n_insert_objects_psql = 100

    def test_bulk_insert_objects_mongo(self):
        objects = generate_random_objects(self.n_insert_objects_mongo)
        self.driver.query("Object", engine="mongo").bulk_insert(objects)
        mongo_objects = self.driver.query("Object", engine="mongo").find_all(paginate=False)
        self.assertIsInstance(mongo_objects, list)
        self.assertEqual(len(mongo_objects), self.n_insert_objects_mongo)

    def test_bulk_insert_detections_mongo(self):
        batch = generate_random_detections(self.n_insert_detections_mongo)
        self.driver.query("Detection", engine="mongo").bulk_insert(batch)
        mongo_detections = self.driver.query("Detection", engine="mongo").find_all(paginate=False)
        self.assertIsInstance(mongo_detections, list)
        self.assertEqual(len(mongo_detections), self.n_insert_detections_mongo)

    def test_bulk_update_objects_mongo(self):
        updated = []
        objects = generate_random_objects(10)
        for obj in objects:
            obj["ndet"] += 1
            obj["oid"] = f"{obj['oid']}UPDATED"
            updated.append(obj["oid"])
        filter_by = [{"_id": f"ALERCE{x}"} for x in range(0, 10)]
        self.driver.query("Object", engine="mongo").bulk_update(objects, filter_by)
        filter_by = {"oid": {"$in": updated}}
        updated_objects = self.driver.query("Object", engine="mongo").find_all(filter_by=filter_by, paginate=False)
        self.assertEqual(len(updated_objects), len(updated))

    def test_find_all_objects_mongo(self):
        mongo_objects = self.driver.query("Object", engine="mongo").find_all(paginate=False)
        self.assertIsInstance(mongo_objects, list)
        self.assertEqual(len(mongo_objects), self.n_insert_objects_mongo)

    def test_find_all_objects_with_filters_mongo(self):
        filter_by = {"aid": {"$in": ["ALERCE1", "ALERCE2"]}}
        mongo_objects = self.driver.query("Object", engine="mongo").find_all(filter_by=filter_by, paginate=False)
        self.assertIsInstance(mongo_objects, list)
        self.assertEqual(len(mongo_objects), 2)

    def test_bulk_insert_objects_psql(self):
        objects = generate_random_objects(self.n_insert_objects_psql)
        self.driver.query("Object", engine="psql").bulk_insert(objects)

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

    def test_find_all_psql(self):
        psql_objects = self.driver.query("Object", engine="psql").find_all(paginate=False)
        self.assertIsInstance(psql_objects, list)
        self.assertEqual(len(psql_objects), 100)

        filter_by = {"aid": {"$in": ["EX0", "EX1"]}}
        psql_objects = self.driver.query("Object", engine="psql").find_all(filter_by=filter_by, paginate=False)
        self.assertIsInstance(psql_objects, list)
        self.assertEqual(len(psql_objects), 2)
