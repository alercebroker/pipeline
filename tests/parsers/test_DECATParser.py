import os
from fastavro import reader
import unittest

from survey_parser_plugins.core import GenericAlert
from survey_parser_plugins.parsers import DECATParser

FILE_PATH = os.path.dirname(__file__)
ATLAS_DATA_PATH = os.path.join(FILE_PATH, "../../notebooks/data/ATLAS_samples")
DECAT_DATA_PATH = os.path.join(FILE_PATH, "../../notebooks/data/DECAT_samples")


def get_content(file_path):
    with open(file_path, "rb") as f:
        content = reader(f).next()
    return content


class TestDECATParser(unittest.TestCase):
    def setUp(self) -> None:
        self._atlas_sample = [get_content(os.path.join(ATLAS_DATA_PATH, f)) for f in os.listdir(ATLAS_DATA_PATH)]
        self._decat_sample = [get_content(os.path.join(DECAT_DATA_PATH, f)) for f in os.listdir(DECAT_DATA_PATH)]

    def test_can_parse(self):
        decat_message = self._decat_sample[0]
        response = DECATParser.can_parse(decat_message)
        self.assertTrue(response)

    def test_cant_parse(self):
        atlas_message = self._atlas_sample[0]
        response = DECATParser.can_parse(atlas_message)
        self.assertFalse(response)

    def test_parse(self):
        decat_message = self._decat_sample[0]
        is_parseable = DECATParser.can_parse(decat_message)
        if is_parseable:
            response = DECATParser.parse_message(decat_message)
            self.assertIsInstance(response, list)
            for i in response:
                self.assertIsInstance(i, GenericAlert)

    def test_bad_message(self):
        atlas_message = self._atlas_sample[0]
        with self.assertRaises(KeyError) as context:
            DECATParser.parse_message(atlas_message)
        self.assertIsInstance(context.exception, Exception)

    def test_get_source(self):
        self.assertEqual(DECATParser.get_source(), "DECAT")
        self.assertNotEqual(DECATParser.get_source(), "DEKAT")
