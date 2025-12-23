import unittest
import os
from fastavro import reader, schema, validate
import pathlib
from schema_parser.parsers.v9_to_v10 import V9ToV10Parser

V9_SCHEMA_PATH = pathlib.Path(__file__).parent.parent.parent.parent.parent / "schemas/surveys/lsst_v9.0/lsst.v9_0.alert.avsc"
V10_SCHEMA_PATH = pathlib.Path(__file__).parent.parent.parent.parent.parent / "schemas/surveys/lsst_v10.0/lsst.v10_0.alert.avsc"
V9_AVRO_PATH = pathlib.Path(__file__).parent.parent / "sample_data/sample_v9.avro"

class TestV9V10Parser(unittest.TestCase):
    def test_v9_to_v10_parse(self):
        
        v9_schema = schema.load_schema(V9_SCHEMA_PATH)
        v10_schema = schema.load_schema(V10_SCHEMA_PATH)

        with open(V9_AVRO_PATH, 'rb') as fo:
            avro_reader = reader(fo, v9_schema)
            for m in avro_reader:
                v9_dict = m

        parser = V9ToV10Parser()
        v10_dict = parser.parse(v9_dict)

        valid = validate(v10_dict, v10_schema)

        self.assertTrue(valid)
