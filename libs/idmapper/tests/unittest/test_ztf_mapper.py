import unittest
import numpy as np
from idmapper.mapper import SURVEY_IDS, catalog_oid_to_masterid, decode_masterid
from idmapper.ztf import is_ztf_oid_valid


class TestZTFMapper(unittest.TestCase):
    def test_max_ztf_oid(self):
        ztf_oid = "ZTF99zzzzzzz"

        masterid = catalog_oid_to_masterid("ZTF", ztf_oid)
        masterid_without_survey = masterid & ((1 << (63 - 8)) - 1)
        year = masterid_without_survey // (26**7)
        sequence = masterid_without_survey % (26**7)

        # Check that the year is 99
        self.assertEqual(
            year,
            99,
            "Master ID should have the correct year bits",
        )

        # Check that the sequence is zzzzzzz
        for i in range(6):
            sequence, remainder = divmod(sequence, 26)
            self.assertEqual(
                remainder,
                ord("z") - ord("a"),
                f"Master ID should have the correct sequence bits at position {i}",
            )

    def test_stored_oid_unchanged(self):
        # A real oid from the multisurvey database. Every stored ZTF oid depends on
        # this encoding, so it must never change.
        self.assertEqual(
            catalog_oid_to_masterid("ZTF", "ZTF17aaaaauj"), 36028933559737489
        )
        self.assertEqual(decode_masterid(36028933559737489), ("ZTF", "ZTF17aaaaauj"))

    def test_ztf_prefix(self):
        self.assertEqual(SURVEY_IDS["ZTF"], 0)
        masterid = catalog_oid_to_masterid("ZTF", "ZTF20abcdefg")
        self.assertEqual(int(masterid) >> (63 - 8), 1)

    def test_non_ztf_rejected(self):
        with self.assertRaises(ValueError):
            catalog_oid_to_masterid("LSST DIA Object", "313853798223069211")
        # An LSST DIA object oid as stored in the database (no ZTF prefix).
        with self.assertRaises(ValueError):
            decode_masterid(313853798223069211)

    def test_masterid_type(self):
        ztf_oid = "ZTF20abcdefg"
        masterid = catalog_oid_to_masterid("ZTF", ztf_oid)
        self.assertIsInstance(masterid, np.int64, "masterid should be of type np.int64")

    def test_is_valid_ztf_oid(self):
        valid_ztf_oids = [
            "ZTF17aalvdaa",
            "ZTF21aabbbba",
            "ZTF00aaccdda",
            "ZTF21aadfggg",
            "ZTF21aahhhha",
        ]
        invalid_ztf_oids = [
            "ZTF21a",
            "ZTF21aabbbbaaa",
            "ZTF9aabbbbaA",
            "ZTF21aabbbba1",
            "ZTF21aabbbba!",
        ]
        for ztf_oid in valid_ztf_oids:
            self.assertTrue(is_ztf_oid_valid(ztf_oid), f"{ztf_oid} should be valid")

        for ztf_oid in invalid_ztf_oids:
            self.assertFalse(is_ztf_oid_valid(ztf_oid), f"{ztf_oid} should be invalid")

    def test_decode_masterid(self):
        ztf_oids = [
            "ZTF17aalvdaa",
            "ZTF21aabbbba",
            "ZTF00aaccdda",
            "ZTF21aadfggg",
            "ZTF21aahhhha",
        ]
        for ztf_oid in ztf_oids:
            # Convert to masterid and back to ztf_oid
            masterid = catalog_oid_to_masterid("ZTF", ztf_oid)
            decoded_ztf_oid = decode_masterid(masterid)[1]
            self.assertEqual(
                ztf_oid,
                decoded_ztf_oid,
                f"Decoded ZTF object ID {decoded_ztf_oid} does not match original {ztf_oid}",
            )

    def test_ztf_conversion_increasing_oids(self):
        i = 0
        for second_to_last_letter in range(26):
            for last_letter in range(26):
                ztf_oid = f"ZTF00aaaaa{chr(ord('a') + second_to_last_letter)}{chr(ord('a') + last_letter)}"
                masterid = catalog_oid_to_masterid("ZTF", ztf_oid)

                # masterid in binary
                masterid_bits = "{:064b}".format(masterid)

                # last 10 bits as a number
                last_10_bits_number = int(masterid_bits[-10:], 2)

                self.assertEqual(
                    last_10_bits_number,
                    i,
                    f"Master ID should have the correct last 10 bits for {ztf_oid}",
                )
                i += 1


if __name__ == "__main__":
    unittest.main()
