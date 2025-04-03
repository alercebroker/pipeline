import unittest
from idmapper.mapper import catalog_oid_to_masterid, decode_masterid


class TestLSSTMapper(unittest.TestCase):
    def test_max_lsst_oid(self):
        lsst_oid = 123456789
        lsst_oid_bits_without_survey = "111010110111100110100010101"

        masterid = catalog_oid_to_masterid("LSST", lsst_oid)
        masterid_without_survey = masterid & ((1 << (63 - 8)) - 1)

        # Check that the LSST object ID is the same as the master ID
        self.assertEqual(
            masterid_without_survey,
            lsst_oid,
            "Master ID should have the correct LSST object ID bits",
        )

        masterid_without_survey_bits = "{0:b}".format(masterid_without_survey)
        self.assertEqual(masterid_without_survey_bits, lsst_oid_bits_without_survey)

    def test_decoder(self):
        lsst_oid = 123456789
        masterid = catalog_oid_to_masterid("LSST", lsst_oid)
        survey, oid = decode_masterid(masterid)

        # Check that the survey is "LSST"
        self.assertEqual(survey, "LSST", "Survey should be LSST")

        # Check that the oid is the same as the original oid
        self.assertEqual(oid, lsst_oid, "OID should be the same as the original OID")
