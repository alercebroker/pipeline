import unittest

from mongo_scribe.mongo.command.decode import decode_message, db_command_factory
from mongo_scribe.mongo.command.exceptions import WrongFormatCommandException
from mongo_scribe.mongo.command.commands import (
    InsertCommand,
    UpdateCommand,
    UpdateProbabilitiesCommand,
)

from mockdata import valid_data_dict, valid_data_json


class DecodeTest(unittest.TestCase):
    def test_decode_message(self):
        decoded = decode_message(valid_data_json)
        self.assertEqual(decoded, valid_data_dict)


# Uses type equals instead of isinstance since there are derived classes
class TestCommandFactory(unittest.TestCase):
    def test_factory_decode_error(self):
        with self.assertRaises(WrongFormatCommandException):
            db_command_factory('{"mock": "val"}')

    def test_factory_with_unknown_type_raises_error(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized command"):
            db_command_factory(
                '{"type": "mock", "data": {}, "collection": "object"}'
            )

    def test_factory_generates_insert(self):
        msg = '{"type": "insert", "data": {"field": "value"}, "collection": "object"}'
        self.assertTrue(type(db_command_factory(msg)) == InsertCommand)

    def test_factory_generates_update(self):
        msg = '{"type": "update", "criteria": {"_id": "id"}, "data": {"field": "value"}, "collection": "object"}'
        self.assertTrue(type(db_command_factory(msg)) == UpdateCommand)

    def test_factory_generates_update_probabilities(self):
        msg = '{"type": "update_probabilities", "criteria": {"_id": "id"}, "data": {"classifier_name": "c", "classifier_version": "1.0"}, "collection": "object"}'
        self.assertTrue(
            type(db_command_factory(msg)) == UpdateProbabilitiesCommand
        )
