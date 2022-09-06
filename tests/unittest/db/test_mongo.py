from pydoc import pager
from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import (
    MongoConnection,
    MongoDatabaseCreator,
    to_camel_case,
)
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
            "USERNAME": "username",
            "PASSWORD": "pwd",
            "PORT": 27017,
            "DATABASE": "database",
        }
        self.conn = MongoConnection(client=self.client, config=self.config)

    def test_to_camel_case(self):
        conf = self.config
        conf["SOME_OTHER_ATTRIBUTE"] = "test"
        new_conf = to_camel_case(conf)
        self.assertDictContainsSubset(
            {"someOtherAttribute": "test", "host": "host"}, new_conf
        )

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
        expected = ["object", "detection", "non_detection", "taxonomy"]
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
                "meanra": 100.0,
                "meandec": 50.0,
                "ndet": "test",
            }
        )
        self.assertIsNotNone(result)
        self.assertTrue(created)

    def test_get_or_create_with_kwargs(self):
        result, created = self.query.get_or_create(
            {
                "aid": "test",
                "oid": "test",
                "firstmjd": "test",
                "lastmjd": "test",
                "meanra": 100.0,
                "meandec": 50.0,
                "ndet": "test",
            },
            _id="test",
        )
        self.assertEqual(result.inserted_id, "test")
        self.assertTrue(created)

    def test_update(self):
        model = Object(
            aid="aid",
            oid="oid",
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet="ndet",
        )
        self.obj_collection.insert_one(model)
        self.query.update(model, {"oid": "edited"})
        f = self.obj_collection.find_one({"oid": "edited"})
        self.assertIsNotNone(f)
        self.assertEqual(self.query.model, Object)

    def test_bulk_update(self):
        model = Object(
            aid="aid",
            oid="oid",
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet="ndet",
        )
        self.obj_collection.insert_one(model)
        self.query.bulk_update([model], [{"oid": "edited"}])
        f = self.obj_collection.find_one({"oid": "edited"})
        self.assertIsNotNone(f)
        # now with filters
        self.query.bulk_update(
            [model], [{"oid": "edited2"}], filter_fields=[{"aid": "aid"}]
        )
        f = self.obj_collection.find_one({"oid": "edited2"})
        self.assertIsNotNone(f)

    def test_bulk_insert(self):
        self.assertEqual(self.obj_collection.count_documents({}), 1)
        self.query.bulk_insert(
            [
                {
                    "aid": "test",
                    "oid": "test",
                    "firstmjd": "test",
                    "lastmjd": "test",
                    "meanra": 100.0,
                    "meandec": 50.0,
                    "ndet": "test",
                }
                for i in range(2)
            ]
        )
        self.assertEqual(self.obj_collection.count_documents({}), 3)

    def test_find_all(self):
        result = self.query.find_all(filter_by={"test": "test"})
        self.assertEqual(result.total, 1)
        self.assertEqual(result.items[0]["test"], "test")

    def test_pagination_without_counting(self):
        self.assertEqual(self.obj_collection.count_documents({}), 1)
        self.query.bulk_insert(
            [
                {
                    "aid": "test",
                    "oid": "test",
                    "firstmjd": "test",
                    "lastmjd": "test",
                    "meanra": 100.0,
                    "meandec": 50.0,
                    "ndet": "test",
                }
                for i in range(2)
            ]
        )
        self.assertEqual(self.obj_collection.count_documents({}), 3)

        paginate = self.query.paginate(page=1, per_page=2, count=False)
        self.assertIsNone(paginate.total)
        self.assertTrue(paginate.has_next)
        self.assertEqual(paginate.next_num, 2)

        paginate = self.query.paginate(page=2, per_page=2, count=False)
        self.assertIsNone(paginate.total)
        self.assertFalse(paginate.has_next)
        self.assertIsNone(paginate.next_num)

    def test_pagination_with_counting(self):
        self.assertEqual(self.obj_collection.count_documents({}), 1)
        self.query.bulk_insert(
            [
                {
                    "aid": "test",
                    "oid": "test",
                    "firstmjd": "test",
                    "lastmjd": "test",
                    "meanra": 100.0,
                    "meandec": 50.0,
                    "ndet": "test",
                }
                for i in range(2)
            ]
        )
        self.assertEqual(self.obj_collection.count_documents({}), 3)

        paginate = self.query.paginate(page=1, per_page=2, count=True)
        self.assertEqual(paginate.total, 3)
        self.assertTrue(paginate.has_next)
        self.assertEqual(paginate.next_num, 2)

        paginate = self.query.paginate(page=2, per_page=2, count=True)
        self.assertEqual(paginate.total, 3)
        self.assertFalse(paginate.has_next)
        self.assertIsNone(paginate.next_num)

    def test_pagination_with_empty_query(self):
        self.assertEqual(self.obj_collection.count_documents({}), 1)
        self.query.bulk_insert(
            [
                {
                    "aid": "test",
                    "oid": "test",
                    "firstmjd": "test",
                    "lastmjd": "test",
                    "meanra": 100.0,
                    "meandec": 50.0,
                    "ndet": "test",
                }
                for i in range(2)
            ]
        )
        self.assertEqual(self.obj_collection.count_documents({}), 3)

        paginate = self.query.paginate({"aid": "fake"}, page=1, per_page=2, count=True)
        self.assertEqual(paginate.total, 0)
        self.assertListEqual(paginate.items, [])

        paginate = self.query.paginate({"aid": "fake"}, page=1, per_page=2, count=False)
        self.assertIsNone(paginate.total)
        self.assertListEqual(paginate.items, [])
