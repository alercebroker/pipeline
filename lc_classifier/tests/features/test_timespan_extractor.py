import unittest
from lc_classifier.features.extractors.timespan_extractor import TimespanExtractor
from lc_classifier.examples.data import get_elasticc_example_2


class TestTimespanExtractor(unittest.TestCase):
    def test_with_elasticc(self):
        astro_object = get_elasticc_example_2()
        feature_extractor = TimespanExtractor()
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)
