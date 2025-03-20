from db_plugins.db.mongo._connection import (
    MongoConnection,
    _MongoConfig,
)
from db_plugins.db.mongo.models import Object
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

    def test_to_camel_case(self):
        conf = self.config
        conf["SOME_OTHER_ATTRIBUTE"] = "test"
        new_conf = _MongoConfig(conf)
        # Replacement for deprecated assertDictContainsSubset
        self.assertEqual(
            new_conf, {**new_conf, **{"someOtherAttribute": "test", "host": "host"}}
        )

    def test_init(self):
        conn = MongoConnection(config=self.config)
        self.assertEqual(
            Object.metadata.database,
            self.config["DATABASE"],
        )
        self.assertEqual(conn.database.name, self.config["DATABASE"])

    @mock.patch("db_plugins.db.mongo._connection.MongoClient")
    def test_create_db(self, mock_mongo):
        mock_mongo.return_value = mongomock.MongoClient()
        conn = MongoConnection(self.config)
        conn.create_db()
        collections = conn.client[self.config["DATABASE"]].list_collection_names()
        expected = [
            "object",
            "detection",
            "forced_photometry",
            "non_detection",
        ]
        self.assertEqual(collections, expected)

    @mock.patch("db_plugins.db.mongo._connection.MongoClient")
    def test_drop_db(self, mock_mongo):
        mock_mongo.return_value = mongomock.MongoClient()
        conn = MongoConnection(self.config)
        db = conn.client[self.config["DATABASE"]]
        db.test.insert_one({"test": "test"})
        databases = conn.client.list_database_names()
        expected = ["database"]
        self.assertEqual(databases, expected)
        conn.drop_db()
        databases = conn.client.list_database_names()
        expected = []
        self.assertEqual(databases, expected)
