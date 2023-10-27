import unittest

from survey_parser_plugins.core import GenericAlert
from survey_parser_plugins.parsers import ATLASParser

from utils import get_content


class TestATLASParser(unittest.TestCase):
    def setUp(self) -> None:
        self._atlas_sample = [get_content("tests/schemas/atlas/alert.avsc")]
        self._ztf_sample = [get_content("tests/schemas/ztf/alert.avsc")]

    def test_can_parse(self):
        atlas_message = self._atlas_sample[0]
        atlas_message["publisher"] = "ATLAS"
        response = ATLASParser.can_parse(atlas_message)
        self.assertTrue(response)

    def test_cant_parse(self):
        ztf_message = self._ztf_sample[0]
        response = ATLASParser.can_parse(ztf_message)
        self.assertFalse(response)

    def test_parse(self):
        atlas_message = self._ztf_sample[0]
        is_parseable = ATLASParser.can_parse(atlas_message)
        if is_parseable:
            response = ATLASParser.parse_message(atlas_message)
            self.assertIsInstance(response, GenericAlert)

    def test_bad_message(self):
        ztf_message = self._ztf_sample[0]
        with self.assertRaises(KeyError) as context:
            ATLASParser.parse_message(ztf_message)
        self.assertIsInstance(context.exception, Exception)

    def test_get_source(self):
        self.assertEqual(ATLASParser.get_source(), "ATLAS")
        self.assertNotEqual(ATLASParser.get_source(), "ATLAZX")
