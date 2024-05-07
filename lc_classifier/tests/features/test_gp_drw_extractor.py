import unittest
from lc_classifier.features.extractors.gp_drw_extractor import GPDRWExtractor
from lc_classifier.examples.data import get_ztf_example, get_elasticc_example_2


class TestGPDRWExtractor(unittest.TestCase):
    def test_with_ztf(self):
        astro_object = get_ztf_example(0)
        feature_extractor = GPDRWExtractor(bands=list("gr"), unit="magnitude")
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_with_elasticc(self):
        astro_object = get_elasticc_example_2()
        feature_extractor = GPDRWExtractor(bands=list("ugrizY"), unit="diff_flux")
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)
