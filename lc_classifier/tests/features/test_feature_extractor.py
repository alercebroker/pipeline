import unittest

from lc_classifier.features.core.base import FeatureExtractor, AstroObject


class TestFeatureExtractor(unittest.TestCase):
    def test_feature_extractor(self):
        class DummyFeatureExtractor(FeatureExtractor):
            def compute_features_single_object(self, astro_object: AstroObject):
                pass

        dummy_feature_extractor = DummyFeatureExtractor()
