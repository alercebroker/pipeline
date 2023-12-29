import unittest

from lc_classifier.features.extractors.tde_extractor import TDEExtractor
from lc_classifier.examples.data import get_ztf_example, get_ztf_forced_training_examples


class TestSPMExtractor(unittest.TestCase):
    def test_ztf(self):
        astro_object = get_ztf_example(2)
        feature_extractor = TDEExtractor(
            bands=list('gr'))
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_ztf_forced_phot(self):
        astro_objects = get_ztf_forced_training_examples()
        feature_extractor = TDEExtractor(
            bands=list('gr'))
        astro_object = astro_objects[0]
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)
