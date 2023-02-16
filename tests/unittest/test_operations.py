from unittest.mock import patch
import unittest
from mockdata import valid_data_dict
from mongo_scribe.db.operations import (
    Operations,
    create_operations,
    execute_operations,
)
from mongo_scribe.command.commands import InsertDbCommand, UpdateDbCommand


@patch("db_plugins.db.mongo.MongoConnection")
@patch("pymongo.collection.Collection")
class OperationTest(unittest.TestCase):
    def setUp(self):
        self.insert_command = InsertDbCommand(
            valid_data_dict["collection"],
            valid_data_dict["type"],
            valid_data_dict["criteria"],
            valid_data_dict["data"],
        )

        self.update_command = UpdateDbCommand(
            valid_data_dict["collection"],
            "update",
            valid_data_dict["criteria"],
            valid_data_dict["data"],
        )

    def test_create_operations_empty(self, _, __):
        operations = create_operations([])
        self.assertEqual(
            operations,
            Operations(
                {"inserts": [], "updates": [], "update_probabilities": []}
            ),
        )

    def test_create_operations(self, _, __):
        operations = create_operations(
            [self.update_command, self.insert_command]
        )

        self.assertEqual(len(operations["inserts"]), 1)
        self.assertEqual(len(operations["updates"]), 1)

    @patch("mongo_scribe.db.operations.get_model_collection")
    def test_execute_operations(
        self, mock_get_collection, mock_collection, mock_connection
    ):
        mock_get_collection.return_value = mock_collection

        operations = create_operations(
            [self.update_command, self.insert_command]
        )

        execute_operations(mock_connection, "object")(operations)
        mock_collection.insert_many.assert_called_with(
            operations["inserts"], ordered=False
        )
        mock_collection.bulk_write.assert_called_with(operations["updates"])

    """
    
    def test_bulk_execute_empty(self, mock_collection, _):
        operations = []

        self.executor.bulk_execute("object", operations)
        mock_collection.insert_many.assert_not_called()
        mock_collection.bulk_write.assert_not_called()

    @patch("mongo_scribe.db.operations.get_model_collection")
    def test_bulk_execute_insert_only(
        self, get_model_coll_mock, mock_collection, _
    ):
        get_model_coll_mock.return_value = mock_collection
        operations = [InsertDbCommand("object", "insert", None, {"field": "value"})]

        self.executor.bulk_execute("object", operations)
        mock_collection.insert_many.assert_called()
        mock_collection.bulk_write.assert_not_called()

    @patch("mongo_scribe.db.operations.get_model_collection")
    def test_bulk_execute_update_only(
        self, get_model_coll_mock, mock_collection, _
    ):
        get_model_coll_mock.return_value = mock_collection
        operations = [
            UpdateDbCommand(
                "object", "update", {"_id": "AID51423"}, {"field": "value"}
            )
        ]

        self.executor.bulk_execute("object", operations)
        mock_collection.insert_many.assert_not_called()
        mock_collection.bulk_write.assert_called()

    @patch("mongo_scribe.db.operations.get_model_collection")
    def test_bulk_execute(self, get_model_coll_mock, mock_collection, _):
        get_model_coll_mock.return_value = mock_collection
        operations = [
            InsertDbCommand("object", "insert", None, {"field": "value"}),
            UpdateDbCommand(
                "object", "update", {"_id": "AID51423"}, {"field": "value"}
            ),
        ]

        self.executor.bulk_execute("object", operations)
        mock_collection.insert_many.assert_called()
        mock_collection.bulk_write.assert_called()
    """
