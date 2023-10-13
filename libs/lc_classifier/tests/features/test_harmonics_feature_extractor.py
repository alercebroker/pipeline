import unittest
from lc_classifier.features.extractors.harmonics_extractor import HarmonicsExtractor
from lc_classifier.examples.data import get_ztf_example


class TestHarmonicsFeatureExtractor(unittest.TestCase):
    def test_ztf_object(self):
        ztf_astro_object = get_ztf_example(1)
        ztf_astro_object.features.loc[0] = [
            'Multiband_period',
            0.5388739,
            'g,r',
            'ztf_survey',
            '1.0.0'
        ]
        feature_extractor = HarmonicsExtractor(
            bands=['g', 'r'],
            use_forced_photo=True
        )
        feature_extractor.compute_features_single_object(ztf_astro_object)
        print(ztf_astro_object.features)
