import unittest
import pathlib
from fastavro import schema, validate
from schema_parser.parsers.v10_to_v11 import V10ToV11Parser

V11_SCHEMA_PATH = (
    pathlib.Path(__file__).parent.parent.parent.parent.parent
    / "schemas/surveys/lsst_v11.0/lsst.v11_0.alert.avsc"
)


def _minimal_dia_source_v10(**overrides):
    """Minimal v10 diaSource; the four int fields that become required in v11 are None."""
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
        # nullable int fields that become required "int" in v11
        "psfNdata": None,
        "trailNdata": None,
        "dipoleNdata": None,
        "bboxSize": None,
    }
    src.update(overrides)
    return src


def _minimal_dia_object_v10(**overrides):
    """Minimal v10 diaObject; the six per-band Ndata fields that become required in v11 are None."""
    obj = {
        "diaObjectId": 1,
        "validityStartMjdTai": 60000.0,
        "ra": 0.0,
        "dec": 0.0,
        "nDiaSources": 1,
        # nullable int fields that become required "int" in v11
        "u_psfFluxNdata": None,
        "g_psfFluxNdata": None,
        "r_psfFluxNdata": None,
        "i_psfFluxNdata": None,
        "z_psfFluxNdata": None,
        "y_psfFluxNdata": None,
    }
    obj.update(overrides)
    return obj


def _alert_v10(dia_source=None, dia_object=None, prv_dia_sources=None):
    return {
        "diaSourceId": 1,
        "observation_reason": None,
        "target_name": None,
        "diaSource": dia_source or _minimal_dia_source_v10(),
        "prvDiaSources": prv_dia_sources,
        "prvDiaForcedSources": None,
        "diaObject": dia_object,
        "ssSource": None,
        "mpc_orbits": None,
        "cutoutDifference": None,
        "cutoutScience": None,
        "cutoutTemplate": None,
    }


class TestV10V11Parser(unittest.TestCase):
    def setUp(self):
        self.v11_schema = schema.load_schema(V11_SCHEMA_PATH)
        self.parser = V10ToV11Parser()

    def test_null_dia_source_int_fields_coerced_to_zero(self):
        v11 = self.parser.parse(_alert_v10())

        self.assertEqual(v11["diaSource"]["psfNdata"], 0)
        self.assertEqual(v11["diaSource"]["trailNdata"], 0)
        self.assertEqual(v11["diaSource"]["dipoleNdata"], 0)
        self.assertEqual(v11["diaSource"]["bboxSize"], 0)
        self.assertTrue(validate(v11, self.v11_schema))

    def test_null_dia_object_int_fields_coerced_to_zero(self):
        v11 = self.parser.parse(_alert_v10(dia_object=_minimal_dia_object_v10()))

        for band in ("u", "g", "r", "i", "z", "y"):
            self.assertEqual(v11["diaObject"][f"{band}_psfFluxNdata"], 0)
        self.assertTrue(validate(v11, self.v11_schema))

    def test_non_null_int_fields_preserved(self):
        src = _minimal_dia_source_v10(psfNdata=42, bboxSize=7)
        v11 = self.parser.parse(_alert_v10(dia_source=src))

        self.assertEqual(v11["diaSource"]["psfNdata"], 42)
        self.assertEqual(v11["diaSource"]["bboxSize"], 7)
        self.assertTrue(validate(v11, self.v11_schema))

    def test_prv_dia_sources_coerced(self):
        v11 = self.parser.parse(
            _alert_v10(prv_dia_sources=[_minimal_dia_source_v10()])
        )

        self.assertEqual(v11["prvDiaSources"][0]["psfNdata"], 0)
        self.assertTrue(validate(v11, self.v11_schema))

    def test_none_dia_object_passes_through(self):
        v11 = self.parser.parse(_alert_v10(dia_object=None))

        self.assertIsNone(v11["diaObject"])
        self.assertTrue(validate(v11, self.v11_schema))
