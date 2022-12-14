from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import (
    MongoConnection,
    MongoDatabaseCreator,
    _MongoConfig
)
from db_plugins.db.mongo.query import MongoQuery, CollectionNotFound
from db_plugins.db.mongo.models import Object, NonDetection
from unittest import mock
import unittest
import mongomock


class MongoConnectionTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "HOST": "host",
            "USERNAME": "username",
            "PASSWORD": "pwd",
            "PORT": 27017,
            "DATABASE": "database",
        }
        self.conn = MongoConnection(config=self.config)

    def test_to_camel_case(self):
        conf = self.config
        conf["SOME_OTHER_ATTRIBUTE"] = "test"
        new_conf = _MongoConfig(conf)
        # Replacement for deprecated assertDictContainsSubset
        self.assertEqual(new_conf, {**new_conf, **{"someOtherAttribute": "test", "host": "host"}})

    def test_factory_method(self):
        conn = new_DBConnection(MongoDatabaseCreator)
        self.assertIsInstance(conn, MongoConnection)

    def test_connect(self):
        self.conn.connect(self.config)
        self.assertEqual(
            Object.metadata.database,
            self.config["DATABASE"],
        )
        self.assertEqual(self.conn.database.name, self.config["DATABASE"])

    @mock.patch('db_plugins.db.mongo.connection.MongoClient')
    def test_create_db(self, mock_mongo):
        mock_mongo.return_value = mongomock.MongoClient()
        self.conn.connect()

        self.conn.create_db()
        collections = self.conn.client[self.config["DATABASE"]].list_collection_names()
        expected = ["object", "detection", "non_detection", "taxonomy", "step", "feature_version", "pipeline"]
        self.assertEqual(collections, expected)

    @mock.patch('db_plugins.db.mongo.connection.MongoClient')
    def test_drop_db(self, mock_mongo):
        mock_mongo.return_value = mongomock.MongoClient()
        self.conn.connect()

        db = self.conn.client[self.config["DATABASE"]]
        db.test.insert_one({"test": "test"})
        databases = self.conn.client.list_database_names()
        expected = ["database"]
        self.assertEqual(databases, expected)
        self.conn.drop_db()
        databases = self.conn.client.list_database_names()
        expected = []
        self.assertEqual(databases, expected)

    @mock.patch('db_plugins.db.mongo.connection.MongoClient')
    def test_query_orm_api_without_model(self, mock_mongo):
        mock_mongo.return_value = mongomock.MongoClient()
        self.conn.connect()

        query = self.conn.query()
        self.assertIsNone(query.collection)
        self.assertIsNone(query.model)

    @mock.patch('db_plugins.db.mongo.connection.MongoClient')
    def test_query_pymongo_api(self, mock_mongo):
        mock_mongo.return_value = mongomock.MongoClient()
        self.conn.connect()

        query = self.conn.query(name="collection")
        self.assertIsInstance(query.collection, mongomock.Collection)
        self.assertIsNone(query.model)

    @mock.patch('db_plugins.db.mongo.connection.MongoClient')
    def test_query_orm_api_with_model(self, mock_mongo):
        mock_mongo.return_value = mongomock.MongoClient()
        self.conn.connect()

        query = self.conn.query(model=Object)
        self.assertIsInstance(query.collection, mongomock.Collection)
        self.assertEqual(query.model, Object)


class MongoQueryTest(unittest.TestCase):
    def setUp(self):
        client = mongomock.MongoClient()
        self.database = client["database"]
        self.obj_collection = self.database["object"]
        self.obj_collection.insert_one({"test": "test"})
        self.query = MongoQuery(
            model=Object,
            database=self.database,
            # _db_store=self.database._store,
        )

    def test_query_initialization_with_collection_name_and_model_fails(self):
        with self.assertRaisesRegex(ValueError, 'Only one of .+ can be defined'):
            MongoQuery(model=Object, name='objects', database=self.database)

    def test_query_initialization_without_collection_name_and_model_fails(self):
        with self.assertRaisesRegex(CollectionNotFound, 'A valid model must be provided'):
            q = MongoQuery(database=self.database)
            q.init_collection()

    def test_check_exists(self):
        self.assertTrue(self.query.check_exists({"test": "test"}))

    def test_get_or_create(self):
        result, created = self.query.get_or_create({"test": "test"})
        self.assertIsNotNone(result)
        self.assertFalse(created)
        result, created = self.query.get_or_create(
            {
                "aid": "test",
                "oid": ["test"],
                "tid": ["test"],
                "corrected": False,
                "stellar": False,
                "sigmara": .1,
                "sigmadec": .2,
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
                "oid": ["test"],
                "tid": ["test"],
                "corrected": False,
                "stellar": False,
                "sigmara": .1,
                "sigmadec": .2,
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
            tid="tid",
            corrected=False,
            stellar=False,
            sigmara=.1,
            sigmadec=.2,
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
        model1 = Object(
            aid="aid1",
            oid=["oid"],
            tid=["tid"],
            corrected=False,
            stellar=False,
            sigmara=.1,
            sigmadec=.2,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet="ndet",
        )
        model2 = Object(
            aid="aid2",
            oid=["oid"],
            tid=["tid"],
            corrected=False,
            stellar=False,
            sigmara=.1,
            sigmadec=.2,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet="ndet",
        )
        self.obj_collection.insert_one(model1)
        self.obj_collection.insert_one(model2)
        self.query.bulk_update([model1, model2], [{"oid": ["edited1"]}, {"oid": ["edited2"]}])
        f = self.obj_collection.find_one({"oid": ["edited1"]})
        self.assertIsNotNone(f)
        self.assertEqual(f["_id"], "aid1")
        f = self.obj_collection.find_one({"oid": ["edited2"]})
        self.assertIsNotNone(f)
        self.assertEqual(f["_id"], "aid2")

    def test_bulk_update_using_filter(self):
        model1 = Object(
            aid="aid1",
            oid=["oid"],
            tid=["tid"],
            corrected=False,
            stellar=False,
            sigmara=.1,
            sigmadec=.2,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet="ndet",
        )
        model2 = Object(
            aid="aid2",
            oid=["oid"],
            tid=["tid"],
            corrected=False,
            stellar=False,
            sigmara=.1,
            sigmadec=.2,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet="ndet",
        )
        self.obj_collection.insert_one(model1)
        self.obj_collection.insert_one(model2)

        self.query.bulk_update(
            [model2], [{"oid": ["edited2"]}], filter_fields=[{"_id": "aid2"}]
        )
        f = self.obj_collection.find_one({"oid": ["edited2"]})
        self.assertIsNotNone(f)
        self.assertEqual(f["_id"], "aid2")

    def test_bulk_update_fails_if_not_all_instances_have_same_model(self):
        model1 = Object(
            aid="aid1",
            oid=["oid"],
            tid=["tid"],
            corrected=False,
            stellar=False,
            sigmara=.1,
            sigmadec=.2,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet="ndet",
        )
        model2 = NonDetection(
            candid="candid",
            aid="aid",
            tid="tid",
            oid="oid",
            mjd=100,
            fid=1,
            diffmaglim=2,
        )
        self.obj_collection.insert_one(model1)
        self.obj_collection.insert_one(model2)
        with self.assertRaisesRegex(TypeError, "All instances"):
            self.query.bulk_update([model1, model2], [{"oid": ["edited1"]}, {"oid": ["edited2"]}])

    def test_bulk_update_fails_if_instances_and_attributes_do_not_match_size(self):
        model1 = Object(
            aid="aid1",
            oid=["oid"],
            tid=["tid"],
            corrected=False,
            stellar=False,
            sigmara=.1,
            sigmadec=.2,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet="ndet",
        )
        model2 = Object(
            aid="aid2",
            oid=["oid"],
            tid=["tid"],
            corrected=False,
            stellar=False,
            sigmara=.1,
            sigmadec=.2,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet="ndet",
        )
        self.obj_collection.insert_one(model1)
        self.obj_collection.insert_one(model2)
        with self.assertRaisesRegex(ValueError, "Length of instances and attributes must match"):
            self.query.bulk_update([model1, model2], [{"oid": ["edited1"]}])

    def test_bulk_insert(self):
        self.assertEqual(self.obj_collection.count_documents({}), 1)
        self.query.bulk_insert(
            [
                {
                    "aid": f"test{i}",
                    "oid": ["test"],
                    "tid": ["test"],
                    "corrected": False,
                    "stellar": False,
                    "sigmara": .1,
                    "sigmadec": .2,
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
                    "aid": f"test{i}",
                    "oid": "test",
                    "tid": "test",
                    "corrected": False,
                    "stellar": False,
                    "sigmara": .1,
                    "sigmadec": .2,
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
                    "aid": f"test{i}",
                    "oid": "test",
                    "tid": "test",
                    "corrected": False,
                    "stellar": False,
                    "sigmara": .1,
                    "sigmadec": .2,
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
                    "aid": f"test{i}",
                    "oid": "test",
                    "tid": "test",
                    "corrected": False,
                    "stellar": False,
                    "sigmara": .1,
                    "sigmadec": .2,
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

        paginate = self.query.paginate({"_id": "fake"}, page=1, per_page=2, count=True)
        self.assertEqual(paginate.total, 0)
        self.assertListEqual(paginate.items, [])

        paginate = self.query.paginate({"_id": "fake"}, page=1, per_page=2, count=False)
        self.assertIsNone(paginate.total)
        self.assertListEqual(paginate.items, [])
