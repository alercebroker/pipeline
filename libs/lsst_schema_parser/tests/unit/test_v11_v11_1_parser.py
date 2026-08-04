import unittest
import pathlib
from fastavro import schema, validate
from schema_parser.parsers.v11_0_to_v11_1 import V11_0ToV11_1Parser

V11_1_SCHEMA_PATH = (
    pathlib.Path(__file__).parent.parent.parent.parent.parent
    / "schemas/surveys/lsst_v11.1/lsst.v11_1.alert.avsc"
)


def _minimal_dia_source_v11_0(**overrides):
    """Minimal v11.0 diaSource; lacks the four fields added in v11.1."""
    src = {
        "diaSourceId": 1,
        "visit": 1,
        "detector": 1,
        "midpointMjdTai": 60000.0,
        "ra": 0.0,
        "dec": 0.0,
        "x": 0.0,
        "y": 0.0,
        "timeProcessedMjdTai": 60000.0,
        # already required "int" since v11.0
        "psfNdata": 0,
        "trailNdata": 0,
        "dipoleNdata": 0,
        "bboxSize": 0,
    }
    src.update(overrides)
    return src


def _alert_v11_0(dia_source=None, dia_object=None, prv_dia_sources=None):
    return {
        "diaSourceId": 1,
        "observation_reason": None,
        "target_name": None,
        "diaSource": dia_source or _minimal_dia_source_v11_0(),
        "prvDiaSources": prv_dia_sources,
        "prvDiaForcedSources": None,
        "diaObject": dia_object,
        "ssSource": None,
        "mpc_orbits": None,
        "cutoutDifference": None,
        "cutoutScience": None,
        "cutoutTemplate": None,
    }


class TestV11_0ToV11_1Parser(unittest.TestCase):
    def setUp(self):
        self.v11_1_schema = schema.load_schema(V11_1_SCHEMA_PATH)
        self.parser = V11_0ToV11_1Parser()

    def test_new_dia_source_fields_added_as_null(self):
        v11_1 = self.parser.parse(_alert_v11_0())

        self.assertIsNone(v11_1["diaSource"]["trailAlgorithm"])
        self.assertIsNone(v11_1["diaSource"]["trail_flag"])
        self.assertIsNone(v11_1["diaSource"]["reliabilityVersion"])
        self.assertIsNone(v11_1["diaSource"]["exposureTime"])
        self.assertTrue(validate(v11_1, self.v11_1_schema))

    def test_existing_new_field_values_preserved(self):
        src = _minimal_dia_source_v11_0(trailAlgorithm=2, exposureTime=30.0)
        v11_1 = self.parser.parse(_alert_v11_0(dia_source=src))

        self.assertEqual(v11_1["diaSource"]["trailAlgorithm"], 2)
        self.assertEqual(v11_1["diaSource"]["exposureTime"], 30.0)
        self.assertTrue(validate(v11_1, self.v11_1_schema))

    def test_prv_dia_sources_get_new_fields(self):
        v11_1 = self.parser.parse(
            _alert_v11_0(prv_dia_sources=[_minimal_dia_source_v11_0()])
        )

        self.assertIsNone(v11_1["prvDiaSources"][0]["trailAlgorithm"])
        self.assertIsNone(v11_1["prvDiaSources"][0]["exposureTime"])
        self.assertTrue(validate(v11_1, self.v11_1_schema))

    def test_none_prv_dia_sources_passes_through(self):
        v11_1 = self.parser.parse(_alert_v11_0(prv_dia_sources=None))

        self.assertIsNone(v11_1["prvDiaSources"])
        self.assertTrue(validate(v11_1, self.v11_1_schema))

    def test_other_records_pass_through(self):
        v11_1 = self.parser.parse(_alert_v11_0(dia_object=None))

        self.assertIsNone(v11_1["diaObject"])
        self.assertTrue(validate(v11_1, self.v11_1_schema))
