import pytest
import unittest
from unittest.mock import patch

from mongo_scribe.db.operations import ScribeDbOperations
from mongo_scribe.command.commands import DbCommand


@patch("db_plugins.db.mongo.MongoConnection")
@patch("pymongo.collection.Collection")
class OperationTest(unittest.TestCase):
    def setUp(self):
        db_config = {
            "MONGO": {
                "DATABASE": "test",
                "PORT": 27017,
                "HOST": "localhost",
                "USERNAME": "user",
                "PASSWORD": "pass",
            }
        }
        operations = ScribeDbOperations(db_config)
        self.db_operations = operations

    def test_bulk_execute_empty(self, mock_collection, _):
        self.db_operations.collection = mock_collection
        operations = []

        self.db_operations.bulk_execute(operations)
        mock_collection.insert_many.assert_not_called()
        mock_collection.bulk_write.assert_not_called()

    def test_bulk_execute_insert_only(self, mock_collection, _):
        self.db_operations.collection = mock_collection
        operations = [DbCommand("insert", None, {"field": "value"})]

        self.db_operations.bulk_execute(operations)
        mock_collection.insert_many.assert_called()
        mock_collection.bulk_write.assert_not_called()

    def test_bulk_execute_update_only(self, mock_collection, _):
        self.db_operations.collection = mock_collection
        operations = [
            DbCommand("update", {"_id": "AID51423"}, {"field": "value"})
        ]

        self.db_operations.bulk_execute(operations)
        mock_collection.insert_many.assert_not_called()
        mock_collection.bulk_write.assert_called()

    def test_bulk_execute(self, mock_collection, _):
        self.db_operations.collection = mock_collection
        operations = [
            DbCommand("insert", None, {"field": "value"}),
            DbCommand("update", {"_id": "AID51423"}, {"field": "value"}),
        ]

        self.db_operations.bulk_execute(operations)
        mock_collection.insert_many.assert_called()
        mock_collection.bulk_write.assert_called()
