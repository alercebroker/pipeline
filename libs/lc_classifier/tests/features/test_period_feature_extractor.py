import unittest
from lc_classifier.features.extractors.period_extractor import PeriodExtractor
from lc_classifier.examples.data import get_ztf_example
from lc_classifier.examples.data import get_elasticc_example
from lc_classifier.examples.data import get_elasticc_example_2


class TestMHPSFeatureExtractor(unittest.TestCase):
    def test_ztf_object(self):
        ztf_astro_object = get_ztf_example()
        feature_extractor = PeriodExtractor(
            bands=['g', 'r'])
        feature_extractor.compute_features_single_object(ztf_astro_object)

        print(ztf_astro_object.features)

    def test_elasticc_object(self):
        elasticc_astro_object = get_elasticc_example()
        feature_extractor = PeriodExtractor(
            bands=[c for c in 'ugrizY'],
            smallest_period=0.045,
            largest_period=50.0,
            trim_lightcurve_to_n_days=500.0,
            min_length=15,
            use_forced_photo=True
        )
        trimmed_lc = feature_extractor.get_observations(elasticc_astro_object)
        trimmed_lc_timespan = trimmed_lc['mjd'].max() - trimmed_lc['mjd'].min()
        print(trimmed_lc_timespan)
        self.assertLessEqual(trimmed_lc_timespan, 500.0)
