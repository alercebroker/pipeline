import unittest
from unittest import mock

from db_plugins.db.mongo.connection import MongoConnection
from db_plugins.db.mongo.models import Object

from sorting_hat_step.utils.database import oid_query, conesearch_query


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_database_connection = mock.create_autospec(MongoConnection)

    def test_oid_query(self):
        # Mock a response with elements in database
        self.mock_database_connection.query(Object).find_one.return_value = {"aid": 1}
        aid = oid_query(self.mock_database_connection)(["x", "y", "z"])
        self.assertEqual(aid, 1)
        self.mock_database_connection.query(Object).find_one.assert_called_with(
            {"oid": {"$in": ["x", "y", "z"]}}
        )

    def test_oid_query_with_no_elements(self):
        self.mock_database_connection.query(Object).find_one.return_value = []
        aid = oid_query(self.mock_database_connection)(["x", "y", "z"])
        self.assertEqual(aid, None)
        self.mock_database_connection.query(Object).find_one.assert_called_with(
            {"oid": {"$in": ["x", "y", "z"]}}
        )

    def test_oid_query_with_response_without_aid_field(self):
        self.mock_database_connection.query(Object).find_one.return_value = {
            "field1": 1,
            "field2": 2,
        }
        with self.assertRaises(KeyError) as context:
            oid_query(self.mock_database_connection)(["x", "y", "z"])
            self.mock_database_connection.query(Object).find_one.assert_called_with(
                {"oid": {"$in": ["x", "y", "z"]}}
            )
        self.assertIsInstance(context.exception, Exception)

    def test_conesearch_query(self):
        self.mock_database_connection.query(Object).collection.find.return_value = [
            {"aid": 1}
        ]
        objects = conesearch_query(self.mock_database_connection)(1, 2, 3)
        assert objects[0]["aid"] == 1
        self.mock_database_connection.query(Object).collection.find.assert_called_with(
            {
                "loc": {
                    "$nearSphere": {
                        "$geometry": {"type": "Point", "coordinates": [1, 2]},
                        "$maxDistance": 3,
                    }
                },
            },
            {"aid": 1},  # only return alerce_id
        )

    def test_conesearch_query_without_results(self):
        self.mock_database_connection.query(Object).collection.find.return_value = []
        objects = conesearch_query(self.mock_database_connection)(1, 2, 3)
        assert len(objects) == 0
        self.mock_database_connection.query(Object).collection.find.assert_called_with(
            {
                "loc": {
                    "$nearSphere": {
                        "$geometry": {"type": "Point", "coordinates": [1, 2]},
                        "$maxDistance": 3,
                    }
                },
            },
            {"aid": 1},  # only return alerce_id
        )
