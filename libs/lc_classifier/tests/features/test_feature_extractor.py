import unittest

from features.core.base import AstroObject
from lc_classifier.features.core.base import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def test_feature_extractor(self):
        class DummyFeatureExtractor(FeatureExtractor):
            def compute_features_single_object(self, astro_object: AstroObject):
                pass

        dummy_feature_extractor = DummyFeatureExtractor()
