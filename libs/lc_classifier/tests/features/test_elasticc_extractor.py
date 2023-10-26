import unittest
from lc_classifier.examples.data import get_elasticc_example
from lc_classifier.features.composites.elasticc import ElasticcFeatureExtractor


class TestElasticcExtractor(unittest.TestCase):
    def test_astro_object(self):
        elasticc_astro_object = get_elasticc_example()
        feature_extractor = ElasticcFeatureExtractor()
        feature_extractor.compute_features_single_object(elasticc_astro_object)
        print(elasticc_astro_object.features)
