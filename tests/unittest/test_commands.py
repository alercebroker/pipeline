import unittest

from mongo_scribe.command.commands import InsertDbCommand, UpdateDbCommand
from mongo_scribe.command.exceptions import (
    NoDataProvidedException,
    UpdateWithNoCriteriaException,
    NoCollectionProvidedException,
)

from mockdata import valid_data_dict


class CommandTests(unittest.TestCase):
    def test_create_dbcommand(self):
        command = InsertDbCommand(
            valid_data_dict["collection"],
            valid_data_dict["type"],
            valid_data_dict["criteria"],
            valid_data_dict["data"],
        )
        self.assertEqual(command.data, valid_data_dict["data"])

    def test_create_dbcommand_no_collection(self):
        self.assertRaises(
            NoCollectionProvidedException,
            lambda: InsertDbCommand(None, "insert", None, valid_data_dict["data"]),
        )

    def test_create_dbcommand_no_data(self):
        self.assertRaises(
            NoDataProvidedException,
            lambda: InsertDbCommand("object", "insert", None, None),
        )

    def test_create_dbcommand_update_without_criteria(self):
        self.assertRaises(
            UpdateWithNoCriteriaException,
            lambda: UpdateDbCommand(
                "object", "update", None, valid_data_dict["data"]
            ),
        )

    def test_insert_dbcommand_get_operation(self):
        insert_command = InsertDbCommand(
            valid_data_dict["collection"],
            valid_data_dict["type"],
            valid_data_dict["criteria"],
            valid_data_dict["data"],
        )
        raw_operation = insert_command.get_raw_operation()
        self.assertIsInstance(raw_operation, dict)

    def test_update_dbcommand_get_operation(self):
        update_command = UpdateDbCommand(
            valid_data_dict["collection"],
            valid_data_dict["type"],
            valid_data_dict["criteria"],
            valid_data_dict["data"],
        )
        raw_operation = update_command.get_raw_operation()
        self.assertEqual(len(raw_operation), 2)

    def test_update_db_command_options(self):
        update_command = UpdateDbCommand(
            valid_data_dict["collection"],
            valid_data_dict["type"],
            valid_data_dict["criteria"],
            valid_data_dict["data"],
            { "upsert": True }
        )
        self.assertEqual(update_command.options.upsert, True)

    def test_update_db_command_default_options(self):
        update_command = UpdateDbCommand(
            valid_data_dict["collection"],
            valid_data_dict["type"],
            valid_data_dict["criteria"],
            valid_data_dict["data"],
        )
        self.assertEqual(update_command.options.upsert, False)