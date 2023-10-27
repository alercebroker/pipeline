import unittest

from survey_parser_plugins.core import ParserSelector, ALeRCEParser
from survey_parser_plugins.parsers import ATLASParser, ZTFParser

from utils import get_content


class TestParserSelector(unittest.TestCase):
    def setUp(self) -> None:
        self._atlas_sample = [get_content("tests/schemas/atlas/alert.avsc")]
        self._atlas_sample[0]["publisher"] = "ATLAS"
        self._ztf_sample = [get_content("tests/schemas/ztf/alert.avsc")]
        self._ztf_sample[0]["publisher"] = "ZTF"
        self._ztf_sample[0]["candidate"]["fid"] = 1

    def test_empty_parser(self):
        parser = ParserSelector()
        self.assertEqual(parser.parsers, set())
        self.assertEqual(parser.__repr__(), str(set()))

    def test_register_parser(self):
        parser = ParserSelector()
        parser.register_parser(ATLASParser)
        self.assertTrue(ATLASParser in parser.parsers)
        self.assertEqual(parser.__repr__(), str({ATLASParser}))

    def test_remove_parser(self):
        parser = ParserSelector()
        parser.register_parser(ATLASParser)
        self.assertTrue(ATLASParser in parser.parsers)
        parser.remove_parser(ATLASParser)
        self.assertEqual(parser.parsers, set())

    def test_parser(self):
        parser = ParserSelector()
        parser.register_parser(ATLASParser)
        parser.register_parser(ZTFParser)

        response_1 = parser.parse(self._ztf_sample)
        response_2 = parser.parse(self._atlas_sample)

        self.assertEqual(len(response_1), len(self._ztf_sample))
        self.assertEqual(len(response_2), len(self._atlas_sample))

        with self.assertRaises(Exception) as context:
            parser.parse([{"bad": "title"}, {"oo": "asdasda"}])

        self.assertIsInstance(context.exception, Exception)


class TestALeRCEParser(unittest.TestCase):
    def setUp(self) -> None:
        self._atlas_sample = [get_content("tests/schemas/atlas/alert.avsc")]
        self._atlas_sample[0]["publisher"] = "ATLAS"
        self._ztf_sample = [get_content("tests/schemas/ztf/alert.avsc")]
        self._ztf_sample[0]["publisher"] = "ZTF"
        self._ztf_sample[0]["candidate"]["fid"] = 1

    def test_init(self):
        parser = ALeRCEParser()
        self.assertTrue(ATLASParser in parser.parsers)
        self.assertTrue(ZTFParser in parser.parsers)

    def test_parser(self):
        parser = ALeRCEParser()

        response_1 = parser.parse(self._ztf_sample)
        response_2 = parser.parse(self._atlas_sample)

        self.assertEqual(len(response_1), len(self._ztf_sample))
        self.assertEqual(len(response_2), len(self._atlas_sample))
