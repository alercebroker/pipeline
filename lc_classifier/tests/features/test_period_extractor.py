import unittest
import pandas as pd
import os
from lc_classifier.features.preprocess import ElasticcPreprocessor
from lc_classifier.features.extractors import PeriodExtractor


class PeriodExtractorTest(unittest.TestCase):
    def setUp(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        lc_filename = os.path.join(
            this_dir,
            'elasticc_lc_38135580.pkl'
        )
        lc = pd.read_pickle(lc_filename)
        preprocessor = ElasticcPreprocessor()
        self.lc = preprocessor.preprocess(lc)

    def test_trim_lightcurve(self):
        period_extractor = PeriodExtractor(
            bands=['u', 'g', 'r', 'i', 'z', 'Y'],
            smallest_period=0.045,
            largest_period=50.0,
            optimal_grid=True,
            trim_lightcurve_to_n_days=500.0,
            min_length=15
        )
        trimmed_lc = period_extractor._trim_lightcurve(self.lc)
        trimmed_lc_timespan = trimmed_lc['time'].max() - trimmed_lc['time'].min()
        print(trimmed_lc_timespan)
        self.assertLessEqual(trimmed_lc_timespan, 500.0)


if __name__ == '__main__':
    unittest.main()
