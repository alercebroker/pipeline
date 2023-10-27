import unittest

from survey_parser_plugins.core import GenericAlert
from survey_parser_plugins.parsers import LSSTParser

from utils import get_content


class TestLSSTParser(unittest.TestCase):
    def setUp(self) -> None:
        self._atlas_sample = [get_content("tests/schemas/atlas/alert.avsc")]
        self._lsst_sample = [
            get_content("tests/schemas/lsst/elasticc.v0_9_1.alert.avsc")
        ]

    def test_can_parse(self):
        lsst_message = self._lsst_sample[0]
        response = LSSTParser.can_parse(lsst_message)
        self.assertTrue(response)

    def test_cant_parse(self):
        atlas_message = self._atlas_sample[0]
        response = LSSTParser.can_parse(atlas_message)
        self.assertFalse(response)

    def test_lsst_psflux_zero(self):
        lsst_msg = self._lsst_sample[0]
        lsst_msg["diaSource"]["psFlux"] = 0.0
        obj = LSSTParser.parse_message(lsst_msg)
        self.assertEqual(obj.isdiffpos, 1)

    def test_parse(self):
        lsst_message = self._lsst_sample[0]
        is_parseable = LSSTParser.can_parse(lsst_message)
        if is_parseable:
            response = LSSTParser.parse_message(lsst_message)
            self.assertIsInstance(response, GenericAlert)
            self.assertIsInstance(response.extra_fields["diaObject"], list)

    def test_parse_many_message(self):
        for lsst_message in self._lsst_sample:
            is_parseable = LSSTParser.can_parse(lsst_message)
            if is_parseable:
                response = LSSTParser.parse_message(lsst_message)
                self.assertIsInstance(response, GenericAlert)

    def test_bad_message(self):
        atlas_message = self._atlas_sample[0]
        with self.assertRaises(KeyError) as context:
            LSSTParser.parse_message(atlas_message)
        self.assertIsInstance(context.exception, Exception)

    def test_get_source(self):
        self.assertEqual(LSSTParser.get_source(), "LSST")
        self.assertNotEqual(LSSTParser.get_source(), "LSSTASTIC")
