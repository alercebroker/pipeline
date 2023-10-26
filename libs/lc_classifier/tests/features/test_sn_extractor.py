import unittest

from lc_classifier.features.extractors.sn_extractor import SNExtractor
from lc_classifier.examples.data import get_elasticc_example, get_ztf_example


class TestSPMExtractor(unittest.TestCase):
    def test_with_elasticc(self):
        astro_object = get_elasticc_example()
        feature_extractor = SNExtractor(
            bands=list('ugrizY'),
            unit='diff_flux',
            use_forced_photo=True
        )
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_ztf_object(self):
        ztf_astro_object = get_ztf_example(2)
        feature_extractor = SNExtractor(
            bands=list('gr'),
            unit='magnitude',
            use_forced_photo=False
        )
        feature_extractor.compute_features_single_object(ztf_astro_object)
        print(ztf_astro_object.features)
