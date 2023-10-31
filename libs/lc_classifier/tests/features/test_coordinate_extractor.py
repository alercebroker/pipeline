import unittest

from lc_classifier.features.extractors.coordinate_extractor import CoordinateExtractor
from lc_classifier.examples.data import get_ztf_example, get_elasticc_example


class TestCoordinateExtractor(unittest.TestCase):
    def test_ztf_object(self):
        astro_object = get_ztf_example(2)
        feature_extractor = CoordinateExtractor()
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_elasticc_object(self):
        astro_object = get_elasticc_example()
        feature_extractor = CoordinateExtractor()
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)
