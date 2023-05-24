import unittest
import pandas as pd
from features.utils.extractor_selector import extractor_selector, ExtractorNotFoundException


class ExtractorSelectorTestCase(unittest.TestCase):

    def test_get_ztf_extractor(self):
        input_str = "ZTF"

        selected_extractor = extractor_selector(input_str)

        self.assertEqual(selected_extractor.NAME, "ztf_lc_features")

    def test_get_elasticc_extractor(self):
        input_str = "ELASTICC"

        selected_extractor = extractor_selector(input_str)

        self.assertEqual(selected_extractor.NAME, "elasticc_lc_features")

    def test_extractor_not_found(self):
        input_str = "ztf"
        with self.assertRaises(ExtractorNotFoundException):
            extractor_selector(input_str)

        input_str = "elasticc"
        with self.assertRaises(ExtractorNotFoundException):
            extractor_selector(input_str)

        input_str = "not_found"
        with self.assertRaises(ExtractorNotFoundException):
            extractor_selector(input_str)
