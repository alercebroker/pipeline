import unittest
import pytest

from mongo_scribe.command.commands import DbCommand
from mongo_scribe.command.exceptions import (
    NoDataProvidedException,
    UpdateWithNoCriteriaException,
    NoCollectionProvidedException
)

from mockdata import valid_data_dict


class CommandTests(unittest.TestCase):
    def test_create_dbcommand(self):
        command = DbCommand(
            valid_data_dict["collection"],
            valid_data_dict["type"],
            valid_data_dict["criteria"],
            valid_data_dict["data"],
        )
        self.assertEqual(command.data, valid_data_dict["data"])

    def test_create_dbcommand_no_collection(self):
        self.assertRaises(
            NoCollectionProvidedException, lambda: DbCommand(None, "insert", None, valid_data_dict["data"])
        )

    def test_create_dbcommand_no_data(self):
        self.assertRaises(
            NoDataProvidedException, lambda: DbCommand("object","insert", None, None)
        )

    def test_create_dbcommand_update_without_criteria(self):
        self.assertRaises(
            UpdateWithNoCriteriaException,
            lambda: DbCommand("object", "update", None, valid_data_dict["data"]),
        )
