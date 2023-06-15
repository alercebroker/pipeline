import math
import unittest
from unittest import mock
from pymongo.errors import BulkWriteError

from db_plugins.db.mongo.connection import MongoConnection
from db_plugins.db.mongo.models import Object

from sorting_hat_step.utils import database


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_db = mock.create_autospec(MongoConnection)

    def test_oid_query(self):
        # Mock a response with elements in database
        self.mock_db.query(Object).collection.find_one.return_value = {"_id": 1}
        aid = database.oid_query(self.mock_db, ["x", "y", "z"])
        self.assertEqual(aid, 1)
        self.mock_db.query(Object).collection.find_one.assert_called_with(
            {"oid": {"$in": ["x", "y", "z"]}}, {"_id": 1}
        )

    def test_oid_query_with_no_elements(self):
        self.mock_db.query(Object).collection.find_one.return_value = None
        aid = database.oid_query(self.mock_db, ["x", "y", "z"])
        self.assertEqual(aid, None)

    def test_oid_query_with_response_without_aid_field(self):
        self.mock_db.query(Object).collection.find_one.return_value = {
            "field1": 1,
            "field2": 2,
        }
        with self.assertRaisesRegex(KeyError, "_id"):
            database.oid_query(self.mock_db, ["x", "y", "z"])

    def test_conesearch_query(self):
        self.mock_db.query(Object).collection.find_one.return_value = {"_id": 1}
        aid = database.conesearch_query(self.mock_db, 180, 0, math.degrees(3600))
        assert aid == 1
        self.mock_db.query(Object).collection.find_one.assert_called_with(
            {
                "loc": {
                    "$nearSphere": {
                        "$geometry": {"type": "Point", "coordinates": [0, 0]},
                        "$maxDistance": 1,
                    },
                },
            },
            {"_id": 1},  # only return alerce_id
        )

    def test_conesearch_query_without_results(self):
        self.mock_db.query(Object).collection.find_one.return_value = None
        aid = database.conesearch_query(self.mock_db, 1, 2, 3)
        assert aid is None

    def test_id_query(self):
        self.mock_db.query(Object).collection.find.return_value = [{"_id": 1, "oid": [10]}, {"_id": 2, "oid": [20, 30]}]
        found = database.id_query(self.mock_db, [1, 2])
        self.assertEqual(found, [{"_id": 1, "oid": [10]}, {"_id": 2, "oid": [20, 30]}])
        self.mock_db.query(Object).collection.find.assert_called_with(
                {"_id": {"$in": [1, 2]}}, {"_id": 1, "oid": 1}
        )

    def test_update_query(self):
        records = [
            {'oid': [10], '_id': 0},
            {'oid': [20, 30], '_id': 1},
                        ]
        database.update_query(self.mock_db, records)

        assert self.mock_db.query(Object).collection.update_one.call_count == 2

        query = {"_id": {"$in": [0]}}
        new_value = { "$set": { 'oid': [10] } }
        self.mock_db.query(Object).collection.update_one.assert_any_call(
            query, new_value, upsert=True
        )
