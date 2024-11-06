import unittest

from lc_classifier.features.extractors.reference_feature_extractor import (
    ReferenceFeatureExtractor,
)
from lc_classifier.examples.data import get_ztf_forced_training_examples


class TestReferenceFeatureExtractor(unittest.TestCase):
    def test_ztf_forced_phot(self):
        astro_objects = get_ztf_forced_training_examples()
        feature_extractor = ReferenceFeatureExtractor(bands=list("gr"))
        astro_object = astro_objects[0]
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)
