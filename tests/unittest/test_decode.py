import unittest
import pytest
from json import loads

from mongo_scribe.command.decode import decode_message
from mongo_scribe.command.exceptions import WrongFormatCommandException

from mockdata import valid_data_dict, valid_data_json


class DecodeTest(unittest.TestCase):
    def test_decode_message(self):
        decoded = decode_message(valid_data_json)
        self.assertEqual(decoded, valid_data_dict)

    def test_decode_message_failure(self):
        self.assertRaises(
            WrongFormatCommandException,
            lambda: decode_message('{ "lol": "men" }'),
        )
