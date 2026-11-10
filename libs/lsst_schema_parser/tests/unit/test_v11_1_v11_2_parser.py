import unittest
import pathlib
from fastavro import schema, validate
from schema_parser.parsers.v11_1_to_v11_2 import V11_1ToV11_2Parser

ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent
V11_2_SCHEMA_PATH = ROOT / "schemas/surveys/lsst_v11.2/lsst.v11_2.alert.avsc"


def _minimal_dia_source_v11_1(**overrides):
    """Minimal v11.1 diaSource; lacks the seven fields added in v11.2."""
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
        "psfNdata": 0,
        "trailNdata": 0,
        "dipoleNdata": 0,
        "bboxSize": 0,
    }
    src.update(overrides)
    return src


def _alert_v11_1(dia_source=None, prv_dia_sources=None):
    return {
        "diaSourceId": 1,
        "observation_reason": None,
        "target_name": None,
        "diaSource": dia_source or _minimal_dia_source_v11_1(),
        "prvDiaSources": prv_dia_sources,
        "prvDiaForcedSources": None,
        "diaObject": None,
        "ssSource": None,
        "mpc_orbits": None,
        "cutoutDifference": None,
        "cutoutScience": None,
        "cutoutTemplate": None,
    }


class TestV11_1ToV11_2Parser(unittest.TestCase):
    def setUp(self):
        self.v11_2_schema = schema.load_schema(V11_2_SCHEMA_PATH)
        self.parser = V11_1ToV11_2Parser()

    def _assert_upgraded(self, source):
        self.assertIsNone(source["pixelFlags_spike"])
        self.assertIsNone(source["templateMjdMean"])
        self.assertIsNone(source["templateMjdMin"])
        self.assertIsNone(source["templateMjdMax"])
        self.assertEqual(source["templateInstrument"], "UNKNOWN")
        self.assertEqual(source["templateVersion"], 0)
        self.assertIs(source["mixedInstruments"], False)

    def test_v11_1_alert_does_not_validate_against_v11_2(self):
        self.assertFalse(
            validate(_alert_v11_1(), self.v11_2_schema, raise_errors=False)
        )

    def test_new_dia_source_fields_added(self):
        v11_2 = self.parser.parse(_alert_v11_1())

        self._assert_upgraded(v11_2["diaSource"])
        self.assertTrue(validate(v11_2, self.v11_2_schema))

    def test_existing_new_field_values_preserved(self):
        src = _minimal_dia_source_v11_1(
            templateInstrument="DECam", templateVersion=1, mixedInstruments=True
        )
        v11_2 = self.parser.parse(_alert_v11_1(dia_source=src))

        self.assertEqual(v11_2["diaSource"]["templateInstrument"], "DECam")
        self.assertEqual(v11_2["diaSource"]["templateVersion"], 1)
        self.assertIs(v11_2["diaSource"]["mixedInstruments"], True)
        self.assertTrue(validate(v11_2, self.v11_2_schema))

    def test_prv_dia_sources_get_new_fields(self):
        v11_2 = self.parser.parse(
            _alert_v11_1(prv_dia_sources=[_minimal_dia_source_v11_1()])
        )

        self._assert_upgraded(v11_2["prvDiaSources"][0])
        self.assertTrue(validate(v11_2, self.v11_2_schema))

    def test_none_prv_dia_sources_passes_through(self):
        v11_2 = self.parser.parse(_alert_v11_1(prv_dia_sources=None))

        self.assertIsNone(v11_2["prvDiaSources"])
        self.assertTrue(validate(v11_2, self.v11_2_schema))
