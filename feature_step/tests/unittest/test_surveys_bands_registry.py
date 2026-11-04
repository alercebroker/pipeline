"""Tests for SurveyBands, the Survey registry and the ztf/lsst band tables."""
import unittest  # test case base class

from features.surveys.lsst import LSSTSurvey  # lsst survey class under test
from features.surveys.registry import get_survey  # registry lookup under test
from features.surveys.ztf import ZTFSurvey  # ztf survey class under test

ZTF_DB = {"g": 1, "r": 2, "g,r": 12, "i": 0, None: 0, "x": 0}  # today's fid_mapper_for_db
ZTF_SUFFIX = {"g": "_1", "r": "_2", "g,r": "_12", "i": "", None: "", "x": ""}  # today's fid_map in parse_output
LSST_DB = {  # today's fid_mapper_for_db_lsst
    "u": 6, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5,  # single bands
    "u,g": 61, "g,r": 12, "r,i": 23, "i,z": 34, "z,y": 45,  # band pairs
    None: 0, "x": 0,  # missing and unknown bands
}  # end of LSST_DB
LSST_SUFFIX = {  # today's suffix_map in parse_output_lsst
    "u": "_6", "g": "_1", "r": "_2", "i": "_3", "z": "_4", "y": "_5",  # single bands
    "u,g": "_61", "g,r": "_12", "r,i": "_23", "i,z": "_34", "z,y": "_45",  # band pairs
    None: "", "x": "",  # missing and unknown bands
}  # end of LSST_SUFFIX


class TestSurveyBandsRegistry(unittest.TestCase):
    """Check band codes, suffixes, letters, registry and ids of both surveys."""

    def test_ztf_bands_match_today(self):
        """ztf db codes and suffixes equal the current parser literals."""
        for band, code in ZTF_DB.items():  # every band in the db table
            self.assertEqual(ZTFSurvey.bands.fid_for_db(band), code)  # db code matches
        for band, suffix in ZTF_SUFFIX.items():  # every band in the suffix table
            self.assertEqual(ZTFSurvey.bands.suffix(band), suffix)  # suffix matches

    def test_lsst_bands_match_today(self):
        """lsst db codes and suffixes equal the current parser literals."""
        for band, code in LSST_DB.items():  # every band in the db table
            self.assertEqual(LSSTSurvey.bands.fid_for_db(band), code)  # db code matches
        for band, suffix in LSST_SUFFIX.items():  # every band in the suffix table
            self.assertEqual(LSSTSurvey.bands.suffix(band), suffix)  # suffix matches

    def test_letter_inverts_singles(self):
        """letter maps a numeric fid back to its band letter."""
        self.assertEqual(LSSTSurvey.bands.letter(6), "u")  # lsst u is 6
        self.assertEqual(ZTFSurvey.bands.letter(3), "i")  # ztf i comes from letters
        self.assertEqual(ZTFSurvey.bands.singles_inverse, {1: "g", 2: "r", 3: "i"})  # ztf band_map_inverse

    def test_registry_has_both(self):
        """get_survey returns instances of the right classes."""
        self.assertIsInstance(get_survey("ztf"), ZTFSurvey)  # ztf lookup
        self.assertIsInstance(get_survey("lsst"), LSSTSurvey)  # lsst lookup

    def test_registry_unknown_raises(self):
        """An unknown survey name raises ValueError listing the known ones."""
        with self.assertRaises(ValueError) as cm:  # unknown name must fail
            get_survey("x")  # lookup of a missing survey
        self.assertIn("ztf", str(cm.exception))  # message lists ztf
        self.assertIn("lsst", str(cm.exception))  # message lists lsst

    def test_sid_tid(self):
        """Survey ids and type ids match the current constants."""
        self.assertEqual((ZTFSurvey.sid, ZTFSurvey.tid), (0, 0))  # ztf ids
        self.assertEqual((LSSTSurvey.sid, LSSTSurvey.tid), (1, 1))  # lsst ids


if __name__ == "__main__":  # run as a script
    unittest.main()  # allow running the file directly
