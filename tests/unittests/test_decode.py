import unittest
import pytest
from json import loads

from mongo_scribe.command.decode import decode_message
from mongo_scribe.command.exceptions import MisformattedCommandExcepction

valid_json = '''
{
    "type": "insert",
    "criteria": {"_id": "AID51423"},
    "data": {"field1": "some_field", "field2": "some_other_field"}
}
'''

valid_data = {
    "type": "insert",
    "criteria": {"_id": "AID51423"},
    "data": {"field1": "some_field", "field2": "some_other_field"},
}

class DecodeTest(unittest.TestCase):
    def test_decode_message(self):
        decoded = decode_message(valid_json)
        self.assertEqual(decoded, valid_data)

    def test_decode_message_failure(self):
        self.assertRaises(
            MisformattedCommandExcepction,
            lambda: decode_message('{ "lol": "men" }')
        )