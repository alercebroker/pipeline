import math
import unittest
from unittest import mock

from sorting_hat_step.database import MongoConnection
from sorting_hat_step.utils import database
from pymongo.database import Database


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_db = mock.create_autospec(MongoConnection)
        self.mock_db.database = mock.create_autospec(Database)

    def test_oid_query(self):
        # Mock a response with elements in database
        self.mock_db.database["object"].find_one.return_value = {"_id": 1, "aid": 1}
        aid = database.oid_query(self.mock_db, ["x", "y", "z"])
        self.assertEqual(aid, 1)
        self.mock_db.database["object"].find_one.assert_called_with(
            {"_id": {"$in": ["x", "y", "z"]}}, {"aid": 1}
        )

    def test_oid_query_with_no_elements(self):
        self.mock_db.database["object"].find_one.return_value = None
        aid = database.oid_query(self.mock_db, ["x", "y", "z"])
        self.assertEqual(aid, None)

    def test_oid_query_with_response_without_aid_field(self):
        self.mock_db.database["object"].find_one.return_value = {
            "field1": 1,
            "field2": 2,
        }
        with self.assertRaisesRegex(KeyError, "aid"):
            database.oid_query(self.mock_db, ["x", "y", "z"])

    def test_conesearch_query(self):
        self.mock_db.database["object"].find_one.return_value = {"_id": 1, "aid": 1}
        aid = database.conesearch_query(
            self.mock_db, 180, 0, math.degrees(3600)
        )
        assert aid == 1
        self.mock_db.database["object"].find_one.assert_called_with(
            {
                "loc": {
                    "$nearSphere": {
                        "$geometry": {"type": "Point", "coordinates": [0, 0]},
                        "$maxDistance": 6.3781e6,
                    },
                },
            },
            {"aid": 1},
        )

    def test_conesearch_query_without_results(self):
        self.mock_db.database["object"].find_one.return_value = None
        aid = database.conesearch_query(self.mock_db, 1, 2, 3)
        assert aid is None

    def test_update_query(self):
        self.mock_db.database["object"].find_one_and_update.side_effect = [
            {"aid": 10, "_id": 0},
            {"aid": 20, "_id": 1},
        ]
        records = [
            {"aid": 10, "_id": 0},
            {"aid": 20, "_id": 1},
        ]

        database.update_query(self.mock_db, records)

        assert (
            self.mock_db.database["object"].find_one_and_update.call_count == 2
        )

        query = {"_id": 0}
        new_value = {
            "$set": {"aid": 10},
        }
        self.mock_db.database["object"].find_one_and_update.assert_any_call(
            query, new_value, upsert=True, return_document=True
        )
