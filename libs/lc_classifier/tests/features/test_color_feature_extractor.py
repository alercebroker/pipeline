import unittest
from lc_classifier.features.extractors.color_feature_extractor import ColorFeatureExtractor
from lc_classifier.examples.data import get_ztf_example
from lc_classifier.examples.data import get_elasticc_example


class TestColorFeatureExtractor(unittest.TestCase):
    def test_ztf_object(self):
        ztf_astro_object = get_ztf_example()
        color_feature_extractor = ColorFeatureExtractor(
            bands=['g', 'r'],
            unit='magnitude'
        )
        color_feature_extractor.compute_features_single_object(ztf_astro_object)

        print(ztf_astro_object.features)

    def test_elasticc_object(self):
        elasticc_astro_object = get_elasticc_example()
        color_feature_extractor = ColorFeatureExtractor(
            bands=[c for c in 'ugrizY'],
            unit='diff_flux'
        )
        color_feature_extractor.compute_features_single_object(elasticc_astro_object)
        print(elasticc_astro_object.features)
