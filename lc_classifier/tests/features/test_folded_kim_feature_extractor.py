import unittest
from lc_classifier.features.extractors.folded_kim_extractor import FoldedKimExtractor
from lc_classifier.examples.data import get_ztf_example
from lc_classifier.examples.data import get_elasticc_example


class TestFoldedKimFeatureExtractor(unittest.TestCase):
    def test_ztf_object(self):
        ztf_astro_object = get_ztf_example(0)
        ztf_astro_object.features.loc[0] = (
            "Multiband_period",
            3.2,
            "g",
            "ztf_survey",
            "1.0.0",
        )
        feature_extractor = FoldedKimExtractor(bands=["g", "r"], unit="magnitude")
        feature_extractor.compute_features_single_object(ztf_astro_object)

        print(ztf_astro_object.features)

    def test_elasticc_object(self):
        elasticc_astro_object = get_elasticc_example()
        elasticc_astro_object.features.loc[0] = (
            "Multiband_period",
            3.2,
            "g",
            "elasticc_survey",
            "1.0.0",
        )
        feature_extractor = FoldedKimExtractor(bands=list("ugrizY"), unit="diff_flux")
        feature_extractor.compute_features_single_object(elasticc_astro_object)
        print(elasticc_astro_object.features)
