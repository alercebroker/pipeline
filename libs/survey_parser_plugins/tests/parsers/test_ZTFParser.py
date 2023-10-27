import unittest

from survey_parser_plugins.core import GenericAlert
from survey_parser_plugins.parsers import ZTFParser
from utils import get_content


class TestZTFParser(unittest.TestCase):
    def setUp(self) -> None:
        self._atlas_sample = [get_content("tests/schemas/atlas/alert.avsc")]
        self._ztf_sample = [get_content("tests/schemas/ztf/alert.avsc")]

    def test_can_parse(self):
        ztf_message = self._ztf_sample[0]
        ztf_message["publisher"] = "ZTF"
        response = ZTFParser.can_parse(ztf_message)
        self.assertTrue(response)

    def test_cant_parse(self):
        atlas_message = self._atlas_sample[0]
        atlas_message["publisher"] = "ATLAS"
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
