from unittest.mock import patch
from itertools import accumulate
import unittest
from mockdata import valid_data_dict
from mongo_scribe.db.factories.update_probability import (
    UpdateProbabilitiesOperation,
)
from mongo_scribe.db.operations import (
    Operations,
    create_operations,
    execute_operations,
)
from mongo_scribe.command.commands import (
    InsertDbCommand,
    UpdateDbCommand,
    UpdateProbabilitiesDbCommand,
)


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

        self.update_probabilites_command = UpdateProbabilitiesDbCommand(
            valid_data_dict["collection"],
            "update_probabilities",
            {"aid": "AID9876"},
            {
                "some_probs": ["prob1", "prob2"],
                "classifier": {
                    "classifier_name": "LC",
                    "classifier_version": "v1",
                },
            },
        )

    def test_create_operations_empty(self, _, __):
        operations = create_operations([])
        self.assertEqual(operations["inserts"], [])
        self.assertEqual(operations["updates"], [])
        self.assertIsInstance(
            operations["update_probabilities"], UpdateProbabilitiesOperation
        )

    def test_create_operations(self, _, __):
        operations = create_operations(
            [
                self.update_command,
                self.insert_command,
                self.update_probabilites_command,
            ]
        )

        self.assertEqual(len(operations["inserts"]), 1)
        self.assertEqual(len(operations["updates"]), 1)
        
        update_probs = operations["update_probabilities"]

        self.assertEqual(len(update_probs.updates), 1)
        self.assertEqual(
            update_probs.classifier,
            {
                "classifier_name": "LC",
                "classifier_version": "v1",
            },
        )

    @patch("mongo_scribe.db.operations.get_model_collection")
    def test_execute_operations(
        self, mock_get_collection, mock_collection, mock_connection
    ):
        mock_get_collection.return_value = mock_collection

        operations = create_operations(
            [
                self.update_command,
                self.insert_command,
                self.update_probabilites_command,
            ]
        )

        execute_operations(mock_connection, "object")(operations)
        mock_collection.insert_many.assert_called_with(
            operations["inserts"], ordered=False
        )
        mock_collection.bulk_write.assert_called_with(operations["updates"])
        mock_collection.update_probabilities.assert_called()

    def test_create_update_probabilites_operation(self, _, __):
        update_probs = UpdateProbabilitiesOperation()
        for command in [
            self.update_probabilites_command,
            self.update_probabilites_command,
        ]:
            update_probs = update_probs.add_update(command)

        self.assertEqual(len(update_probs.updates), 2)
