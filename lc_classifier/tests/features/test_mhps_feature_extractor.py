import unittest
from lc_classifier.features.extractors.mhps_extractor import MHPSExtractor
from lc_classifier.examples.data import get_ztf_example
from lc_classifier.examples.data import get_elasticc_example


class TestMHPSFeatureExtractor(unittest.TestCase):
    def test_ztf_object(self):
        ztf_astro_object = get_ztf_example(0)
        feature_extractor = MHPSExtractor(bands=["g", "r"], unit="magnitude")
        feature_extractor.compute_features_single_object(ztf_astro_object)

        print(ztf_astro_object.features)

    def test_elasticc_object(self):
        elasticc_astro_object = get_elasticc_example()
        feature_extractor = MHPSExtractor(bands=list("ugrizY"), unit="diff_flux")
        feature_extractor.compute_features_single_object(elasticc_astro_object)
        print(elasticc_astro_object.features)
