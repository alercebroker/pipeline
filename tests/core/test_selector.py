import os
from fastavro import reader
import unittest

from survey_parser_plugins.core import ParserSelector, ALeRCEParser
from survey_parser_plugins.parsers import ATLASParser, ZTFParser

FILE_PATH = os.path.dirname(__file__)
ATLAS_DATA_PATH = os.path.join(FILE_PATH, "../../notebooks/data/ATLAS_samples")
DECAT_DATA_PATH = os.path.join(FILE_PATH, "../../notebooks/data/DECAT_samples")
ZTF_DATA_PATH = os.path.join(FILE_PATH, "../../notebooks/data/ZTF_samples")


def get_content(file_path):
    with open(file_path, "rb") as f:
        content = reader(f).next()
    return content


class TestParserSelector(unittest.TestCase):
    def setUp(self) -> None:
        self._atlas_sample = [get_content(os.path.join(ATLAS_DATA_PATH, f)) for f in os.listdir(ATLAS_DATA_PATH)]
        self._ztf_sample = [get_content(os.path.join(ZTF_DATA_PATH, f)) for f in os.listdir(ZTF_DATA_PATH)]

    def test_empty_parser(self):
        parser = ParserSelector(extra_fields=False)
        self.assertEqual(parser.parsers, set())

    def test_register_parser(self):
        parser = ParserSelector(extra_fields=False)
        parser.register_parser(ATLASParser)
        self.assertTrue(ATLASParser in parser.parsers)

    def test_remove_parser(self):
        parser = ParserSelector(extra_fields=False)
        parser.register_parser(ATLASParser)
        self.assertTrue(ATLASParser in parser.parsers)
        parser.remove_parser(ATLASParser)
        self.assertEqual(parser.parsers, set())

    def test_parser(self):
        parser = ParserSelector(extra_fields=False)
        parser.register_parser(ATLASParser)
        parser.register_parser(ZTFParser)

        response_1 = parser.parse(self._ztf_sample)
        response_2 = parser.parse(self._atlas_sample)

        self.assertEqual(len(response_1), len(self._ztf_sample))
        self.assertEqual(len(response_2), len(self._atlas_sample))


class TestALeRCEParser(unittest.TestCase):
    def setUp(self) -> None:
        self._atlas_sample = [get_content(os.path.join(ATLAS_DATA_PATH, f)) for f in os.listdir(ATLAS_DATA_PATH)]
        self._decat_sample = [get_content(os.path.join(DECAT_DATA_PATH, f)) for f in os.listdir(DECAT_DATA_PATH)]
        self._ztf_sample = [get_content(os.path.join(ZTF_DATA_PATH, f)) for f in os.listdir(ZTF_DATA_PATH)]

    def test_init(self):
        parser = ALeRCEParser(extra_fields=False)
        self.assertTrue(ATLASParser in parser.parsers)
        self.assertTrue(ZTFParser in parser.parsers)

    def test_parser(self):
        parser = ALeRCEParser(extra_fields=False)

        response_1 = parser.parse(self._ztf_sample)
        response_2 = parser.parse(self._decat_sample)
        response_3 = parser.parse(self._atlas_sample)

        self.assertEqual(len(response_1), len(self._ztf_sample))
        self.assertEqual(len(response_2), len(self._decat_sample))
        self.assertEqual(len(response_3), len(self._atlas_sample))
