import unittest

from lc_classifier.features.extractors.ulens_extractor import MicroLensExtractor
from lc_classifier.examples.data import get_ztf_forced_training_examples


class TestMicroLensExtractor(unittest.TestCase):
    def test_ztf_forced_phot(self):
        astro_objects = get_ztf_forced_training_examples()
        feature_extractor = MicroLensExtractor(bands=list("gr"))
        astro_object = astro_objects[0]
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)
