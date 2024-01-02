import unittest
import pandas as pd
from features.utils.selector import (
    selector,
    ExtractorNotFoundException,
)


class ExtractorSelectorTestCase(unittest.TestCase):
    def test_get_ztf_extractor(self):
        input_str = "ZTF"
        selected_extractor = selector(input_str)
        self.assertEqual(selected_extractor.NAME, "ztf_lc_features")

        input_str = "ztf"
        selected_extractor = selector(input_str)
        self.assertEqual(selected_extractor.NAME, "ztf_lc_features")

    def test_get_elasticc_extractor(self):
        input_str = "ELASTICC"
        selected_extractor = selector(input_str)
        self.assertEqual(selected_extractor.NAME, "elasticc_lc_features")

        input_str = "ELAsTiCC"
        selected_extractor = selector(input_str)
        self.assertEqual(selected_extractor.NAME, "elasticc_lc_features")

    def test_get_elasticc_extractor(self):
        input_str = "atlas"
        selected_extractor = selector(input_str)
        self.assertEqual(selected_extractor.NAME, "atlas_lc_features")

        input_str = "ATLAS"
        selected_extractor = selector(input_str)
        self.assertEqual(selected_extractor.NAME, "atlas_lc_features")

    def test_extractor_not_found(self):
        input_str = "dummy"
        with self.assertRaises(ExtractorNotFoundException):
            selector(input_str)
