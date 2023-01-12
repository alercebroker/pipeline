import unittest
import pytest

from mongo_scribe.command.commands import DbCommand
from mongo_scribe.command.exceptions import (
    NoDataProvidedException,
    UpdateWithNoCriteriaException,
)

dummy_data = {
    "type": "insert",
    "criteria": {"_id": "AID51423"},
    "data": {"field1": "some_field", "field2": "some_other_field"},
}


class CommandTests(unittest.TestCase):
    def test_create_dbcommand(self):
        command = DbCommand(
            dummy_data["type"], dummy_data["criteria"], dummy_data["data"]
        )
        self.assertEqual(command.data, dummy_data["data"])

    def test_create_dbcommand_no_data(self):
        self.assertRaises(
            NoDataProvidedException, lambda: DbCommand("insert", None, None)
        )

    def test_create_dbcommand_update_without_criteria(self):
        self.assertRaises(
            UpdateWithNoCriteriaException,
            lambda: DbCommand("update", None, dummy_data["data"]),
        )
