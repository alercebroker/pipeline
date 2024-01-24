import os
from fastavro import reader
import unittest

from survey_parser_plugins.core import GenericAlert
from survey_parser_plugins.parsers import ZTFParser

FILE_PATH = os.path.dirname(__file__)
ATLAS_DATA_PATH = os.path.join(FILE_PATH, "../data/ATLAS_samples")
ZTF_DATA_PATH = os.path.join(FILE_PATH, "../data/ZTF_samples")


def get_content(file_path):
    with open(file_path, "rb") as f:
        for content in reader(f):
            return content


class TestZTFParser(unittest.TestCase):
    def setUp(self) -> None:
        self._atlas_sample = [
            get_content(os.path.join(ATLAS_DATA_PATH, f))
            for f in os.listdir(ATLAS_DATA_PATH)
        ]
        self._ztf_sample = [
            get_content(os.path.join(ZTF_DATA_PATH, f))
            for f in os.listdir(ZTF_DATA_PATH)
        ]

    def test_can_parse(self):
        ztf_message = self._ztf_sample[0]
        response = ZTFParser.can_parse(ztf_message)
        self.assertTrue(response)

    def test_cant_parse(self):
        atlas_message = self._atlas_sample[0]
        response = ZTFParser.can_parse(atlas_message)
        self.assertFalse(response)

    def test_parse(self):
        ztf_message = self._ztf_sample[0]
        is_parseable = ZTFParser.can_parse(ztf_message)
        if is_parseable:
            response = ZTFParser.parse_message(ztf_message)
            self.assertIsInstance(response, GenericAlert)

    def test_bad_message(self):
        atlas_message = self._atlas_sample[0]
        with self.assertRaises(KeyError) as context:
            ZTFParser.parse_message(atlas_message)
        self.assertIsInstance(context.exception, Exception)

    def test_get_source(self):
        self.assertEqual(ZTFParser.get_source(), "ZTF")
        self.assertNotEqual(ZTFParser.get_source(), "STFF")
