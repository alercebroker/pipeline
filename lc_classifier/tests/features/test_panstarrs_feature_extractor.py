import unittest
from lc_classifier.features.extractors.panstarrs_feature_extractor import (
    PanStarrsFeatureExtractor,
)
from lc_classifier.examples.data import get_ztf_forced_phot_cepheid


class TestPanStarrsFeatureExtractor(unittest.TestCase):
    def test_ztf_forced_phot(self):
        astro_object = get_ztf_forced_phot_cepheid()
        feature_extractor = PanStarrsFeatureExtractor()
        feature_extractor.compute_features_single_object(astro_object)
        features = astro_object.features
        print(features)
