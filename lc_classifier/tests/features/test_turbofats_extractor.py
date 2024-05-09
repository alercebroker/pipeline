import unittest
from lc_classifier.features.extractors.turbofats_extractor import TurboFatsExtractor
from lc_classifier.examples.data import get_ztf_example
from lc_classifier.examples.data import get_elasticc_example_2


class TestPeriodFeatureExtractor(unittest.TestCase):
    def test_ztf_object(self):
        ztf_astro_object = get_ztf_example(1)
        feature_extractor = TurboFatsExtractor(["g", "r"], unit="magnitude")
        feature_extractor.compute_features_single_object(ztf_astro_object)
        print(ztf_astro_object.features)

    def test_elasticc_object(self):
        elasticc_astro_object = get_elasticc_example_2()
        feature_extractor = TurboFatsExtractor(["g", "r"], unit="diff_flux")
        feature_extractor.compute_features_single_object(elasticc_astro_object)
        print(elasticc_astro_object.features)
