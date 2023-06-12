import unittest

from unittest import mock
from ingestion_step.utils.multi_driver.connection import MultiDriverConnection
from ingestion_step.utils.multi_driver.query import filter_to_psql, update_to_psql
from db_plugins.db.sql.models import Object
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList
from data.messages import generate_random_objects

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


class MultiDriverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = MultiDriverConnection(CONFIG)
        cls.n_insert_objects_mongo = 100
        cls.n_insert_detections_mongo = 100
        cls.n_insert_objects_psql = 100

    @mock.patch("db_plugins.db.sql.SQLConnection.connect")
    @mock.patch("db_plugins.db.mongo.MongoConnection.connect")
    def test_connect(self, mongo_driver: mock.Mock, psql_driver: mock.Mock):
        self.driver.connect()
        mongo_driver.assert_called()
        psql_driver.assert_called()

    @mock.patch("db_plugins.db.sql.SQLConnection.create_db")
    @mock.patch("db_plugins.db.mongo.MongoConnection.create_db")
    def test_create_db(self, mongo_driver: mock.Mock, psql_driver: mock.Mock):
        self.driver.create_db()
        mongo_driver.assert_called()
        psql_driver.assert_called()

    @mock.patch("db_plugins.db.sql.SQLConnection.drop_db")
    @mock.patch("db_plugins.db.mongo.MongoConnection.drop_db")
    def test_drop_db(self, mongo_driver: mock.Mock, psql_driver: mock.Mock):
        self.driver.psql_driver.session = mock.Mock()
        self.driver.drop_db()
        mongo_driver.assert_called()
        psql_driver.assert_called()

    @mock.patch("db_plugins.db.sql.SQLConnection.query")
    def test_query_find_all_psql(self, psql_driver: mock.Mock):
        psql_driver.find_all.return_value = []
        response = self.driver.query("Detection", engine="psql").find_all({})
        self.assertIsInstance(response, list)
        self.assertListEqual(response, [])

    @mock.patch("db_plugins.db.sql.SQLConnection.query")
    def test_bulk_insert_psql(self, psql_driver: mock.Mock):
        objects = generate_random_objects(10)
        self.driver.query("Object", engine="psql").bulk_insert(objects)
        self.assertTrue(psql_driver.called)

    def test_bulk_update_psql(self):
        objects = generate_random_objects(10)
        filter_by = [{"_id": x["oid"]} for x in objects]
        self.driver.psql_driver.engine = mock.Mock()
        self.driver.query("Object", engine="psql").bulk_update(objects, filter_by=filter_by)
        calls = self.driver.psql_driver.engine.mock_calls
        self.assertEqual(len(calls), 1)

    def test_query_find_all_wrong_model(self):
        with self.assertRaises(Exception) as e:
            self.driver.query("Student", engine="psql").find_all({})
        self.assertIsInstance(e.exception, Exception)

    def test_query_find_all_wrong_engine(self):
        with self.assertRaises(Exception) as e:
            self.driver.query("Detection", engine="cassandra").find_all({})
        self.assertIsInstance(e.exception, Exception)

    @mock.patch("db_plugins.db.mongo.MongoConnection.query")
    def test_query_find_all_mongo(self, mongo_driver: mock.Mock):
        mongo_driver.find_all.return_value = []
        response = self.driver.query("Detection", engine="mongo").find_all({})
        self.assertIsInstance(response, list)
        self.assertListEqual(response, [])

    @mock.patch("db_plugins.db.mongo.MongoConnection.query")
    def test_bulk_insert_mongo(self, mongo_driver: mock.Mock):
        objects = generate_random_objects(10)
        self.driver.query("Object").bulk_insert(objects)
        self.assertTrue(mongo_driver.called)

    @mock.patch("db_plugins.db.mongo.MongoConnection.query")
    def test_bulk_update_mongo(self, mongo_driver: mock.Mock):
        objects = generate_random_objects(10)
        filter_by = [{"_id": x["aid"]} for x in objects]
        self.driver.query("Object").bulk_update(objects, filter_by=filter_by)
        self.assertTrue(mongo_driver.called)

    def test_query_find_one(self):
        with self.assertRaises(NotImplementedError) as e:
            self.driver.query("Detection").find_one()
        self.assertIsInstance(e.exception, NotImplementedError)

    def test_paginate(self):
        with self.assertRaises(NotImplementedError) as e:
            self.driver.query("Detection").paginate()
        self.assertIsInstance(e.exception, NotImplementedError)

    def test_check_exists(self):
        with self.assertRaises(NotImplementedError) as e:
            self.driver.query("Detection").check_exists(None, {})
        self.assertIsInstance(e.exception, NotImplementedError)

    def test_get_or_create(self):
        with self.assertRaises(NotImplementedError) as e:
            self.driver.query("Detection").get_or_create(None, {})
        self.assertIsInstance(e.exception, NotImplementedError)

    def test_update(self):
        with self.assertRaises(NotImplementedError) as e:
            self.driver.query("Detection").update(None, {})
        self.assertIsInstance(e.exception, NotImplementedError)

    def test_filter_to_psql(self):
        filter_by = {"aid": {"$in": ["ZTF1", "ATLAS1", "BART1"]}}
        psql_filter = filter_to_psql(Object, filter_by)
        self.assertIsInstance(psql_filter, BinaryExpression)

        filter_by = {"aid": {"$in": ["ZTF1", "ATLAS1", "BART1"]}, "firstmjd": 10}
        psql_filter = filter_to_psql(Object, filter_by)
        self.assertIsInstance(psql_filter, BooleanClauseList)

        filter_by = {}
        psql_filter = filter_to_psql(Object, filter_by)
        self.assertIsInstance(psql_filter, dict)

        with self.assertRaises(AttributeError) as e:
            filter_by = {"attribute_that_no_exists": {"$in": ["ZTF1", "ATLAS1", "BART1"]}}
            filter_to_psql(Object, filter_by)
        self.assertIsInstance(e.exception, AttributeError)

    def test_update_filters_to_psql(self):
        filter_by = [{"_id": "1"}, {"_id": "2"}]
        psql_filter = update_to_psql(Object, filter_by)
        self.assertIsInstance(psql_filter, BinaryExpression)

        filter_by = [{"_id": "1", "firstmjd": 1}, {"_id": "2", "firstmjd": 2}]
        psql_filter = update_to_psql(Object, filter_by)
        self.assertIsInstance(psql_filter, BooleanClauseList)

        filter_by = []
        psql_filter = update_to_psql(Object, filter_by)
        self.assertIsInstance(psql_filter, dict)

        with self.assertRaises(AttributeError) as e:
            filter_by = [{"_id": "1", "mega": 10}, {"_id": "2", "mega": 10}]
            update_to_psql(Object, filter_by)
        self.assertIsInstance(e.exception, AttributeError)
