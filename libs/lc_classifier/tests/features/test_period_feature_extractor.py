import unittest
from lc_classifier.features.extractors.period_extractor import PeriodExtractor
from lc_classifier.examples.data import get_ztf_example
from lc_classifier.examples.data import get_elasticc_example_2
import numpy as np
import pandas as pd


class TestPeriodFeatureExtractor(unittest.TestCase):
    def test_ztf_object(self):
        ztf_astro_object = get_ztf_example(1)
        feature_extractor = PeriodExtractor(
            bands=['g', 'r'],
            smallest_period=0.045,
            largest_period=500.0,
            trim_lightcurve_to_n_days=1500.0,
            min_length=15,
            use_forced_photo=False
        )
        feature_extractor.compute_features_single_object(ztf_astro_object)
        period = ztf_astro_object.features[ztf_astro_object.features['name'] == 'Multiband_period']
        period = period['value'].values[0]
        self.assertLessEqual(np.abs(0.5388 - period), 0.01)

    def test_trim_with_elasticc(self):
        elasticc_astro_object = get_elasticc_example_2()
        feature_extractor = PeriodExtractor(
            bands=list('ugrizY'),
            smallest_period=0.045,
            largest_period=50.0,
            trim_lightcurve_to_n_days=500.0,
            min_length=15,
            use_forced_photo=True
        )
        all_obs = pd.concat([
            elasticc_astro_object.detections,
            elasticc_astro_object.forced_photometry
        ], axis=0)
        original_timespan = all_obs['mjd'].max() - all_obs['mjd'].min()
        self.assertGreater(original_timespan, 500.0)
        trimmed_lc = feature_extractor.get_observations(elasticc_astro_object)
        trimmed_lc_timespan = trimmed_lc['mjd'].max() - trimmed_lc['mjd'].min()
        self.assertLessEqual(trimmed_lc_timespan, 500.0)
