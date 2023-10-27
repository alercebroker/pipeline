import os
from fastavro.schema import load_schema
from fastavro.utils import generate_many
import unittest

from survey_parser_plugins.core import GenericAlert
from survey_parser_plugins.parsers import ZTFParser


def get_content(schema_path):
    schema = load_schema(schema_path)
    generator = generate_many(schema, size=10)
    return [x for x in generator]


class TestZTFParser(unittest.TestCase):
    def setUp(self) -> None:
        self._atlas_sample = [get_content("tests/data/atlas_schema.avsc")]
        self._ztf_sample = [get_content("tests/schemas/ztf/alert.avsc")]

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
