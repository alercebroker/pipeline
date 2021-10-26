from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import MongoConnection, MongoDatabaseCreator
from db_plugins.db.mongo.query import mongo_query_creator
from db_plugins.db.mongo.models import Object
import unittest
import mongomock


class MongoConnectionTest(unittest.TestCase):
    def setUp(self):
        self.query_class = mongomock.collection.Collection
        self.client = mongomock.MongoClient()
        self.config = {
            "HOST": "host",
            "USER": "username",
            "PASSWORD": "pwd",
            "PORT": 27017,
            "DATABASE": "database",
        }
        self.conn = MongoConnection(client=self.client, config=self.config)

    def test_factory_method(self):
        conn = new_DBConnection(MongoDatabaseCreator)
        self.assertIsInstance(conn, MongoConnection)

    def test_connect(self):
        self.conn.connect(self.config)
        self.assertEqual(self.conn.config, self.config)
        self.assertEqual(
            self.conn.base.metadata.database,
            self.config["DATABASE"],
        )
        self.assertEqual(self.conn.database.name, self.config["DATABASE"])

    def test_create_db(self):
        self.conn.create_db()
        collections = self.client[self.config["DATABASE"]].list_collection_names()
        expected = ["object", "detection", "non_detection"]
        self.assertEqual(collections, expected)

    def test_drop_db(self):
        db = self.client[self.config["DATABASE"]]
        db.test.insert_one({"test": "test"})
        databases = self.client.list_database_names()
        expected = ["database"]
        self.assertEqual(databases, expected)
        self.conn.drop_db()
        databases = self.client.list_database_names()
        expected = []
        self.assertEqual(databases, expected)

    def test_query(self):
        self.conn.database = self.client.database
        query = self.conn.query(self.query_class)
        self.assertIsInstance(query, self.query_class)
        self.assertIsInstance(query, mongomock.collection.Collection)

    def test_query_pymongo_api(self):
        self.conn.database = self.client.database
        query = self.conn.query(
            self.query_class,
            name="collection",
            _db_store=self.conn.database._store,
        )
        self.assertIsInstance(query, self.query_class)
        self.assertIsInstance(query, mongomock.collection.Collection)
        self.assertIsNone(query.model)

    def test_query_orm_api_with_model(self):
        self.conn.database = self.client.database
        query = self.conn.query(
            self.query_class,
            model=Object,
            _db_store=self.conn.database._store,
        )
        self.assertIsInstance(query, self.query_class)
        self.assertIsInstance(query, mongomock.collection.Collection)
        self.assertEqual(query.model, Object)

    def test_query_orm_api_without_model(self):
        self.conn.database = self.client.database
        query = self.conn.query(self.query_class)
        self.assertIsNone(query.model)


class MongoQueryTest(unittest.TestCase):
    def setUp(self):
        client = mongomock.MongoClient()
        self.database = client["database"]
        self.obj_collection = self.database["object"]
        self.obj_collection.insert_one({"test": "test"})
        self.mongo_query_class = mongo_query_creator(mongomock.collection.Collection)
        self.query = self.mongo_query_class(
            model=Object,
            database=self.database,
            _db_store=self.database._store,
        )

    def test_check_exists(self):
        self.assertTrue(self.query.check_exists({"test": "test"}))

    def test_get_or_create(self):
        result, created = self.query.get_or_create({"test": "test"})
        self.assertIsNotNone(result)
        self.assertFalse(created)
        result, created = self.query.get_or_create(
            {
                "aid": "test",
                "oid": "test",
                "firstmjd": "test",
                "lastmjd": "test",
                "meanra": "test",
                "meandec": "test",
                "ndet": "test"
            }
        )
        self.assertIsNotNone(result)
        self.assertTrue(created)

    def test_update(self):
        self.query.update({"test": "test"}, {"$set": {"test": "edited"}})
        f = self.obj_collection.find_one({"test": "edited"})
        self.assertIsNotNone(f)

    def test_bulk_update(self):
        self.assertEqual(self.obj_collection.count_documents({}), 1)
        self.query.bulk_insert(
            [
                {
                    "aid": "test",
                    "oid": "test",
                    "firstmjd": "test",
                    "lastmjd": "test",
                    "meanra": "test",
                    "meandec": "test",
                    "ndet": "test"
                }
                for i in range(2)
            ]
        )
        self.assertEqual(self.obj_collection.count_documents({}), 3)
