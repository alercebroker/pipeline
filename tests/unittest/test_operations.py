import pytest
import unittest
from unittest.mock import patch, Mock
from mongo_scribe.db.models import get_model_collection
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
        operations = []

        self.db_operations.bulk_execute(operations)
        mock_collection.insert_many.assert_not_called()
        mock_collection.bulk_write.assert_not_called()

    @patch('mongo_scribe.db.operations.get_model_collection')
    def test_bulk_execute_insert_only(self, get_model_coll_mock, mock_collection, _):
        get_model_coll_mock.return_value = mock_collection
        operations = [DbCommand("object", "insert", None, {"field": "value"})]

        self.db_operations.bulk_execute(operations)
        mock_collection.insert_many.assert_called()
        mock_collection.bulk_write.assert_not_called()

    @patch('mongo_scribe.db.operations.get_model_collection')
    def test_bulk_execute_update_only(self, get_model_coll_mock, mock_collection, _):
        get_model_coll_mock.return_value = mock_collection
        operations = [
            DbCommand("object", "update", {"_id": "AID51423"}, {"field": "value"})
        ]

        self.db_operations.bulk_execute(operations)
        mock_collection.insert_many.assert_not_called()
        mock_collection.bulk_write.assert_called()

    @patch('mongo_scribe.db.operations.get_model_collection')
    def test_bulk_execute(self, get_model_coll_mock, mock_collection, _):
        get_model_coll_mock.return_value = mock_collection
        operations = [
            DbCommand("object", "insert", None, {"field": "value"}),
            DbCommand("object", "update", {"_id": "AID51423"}, {"field": "value"}),
        ]

        self.db_operations.bulk_execute(operations)
        mock_collection.insert_many.assert_called()
        mock_collection.bulk_write.assert_called()
