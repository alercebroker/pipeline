import unittest

from lc_classifier.examples.data import get_ztf_example
from lc_classifier.utils import plot_astro_object
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
from lc_classifier.features.extractors.harmonics_extractor import HarmonicsExtractor
import numpy as np


class TestZTFPreprocessor(unittest.TestCase):
    def test_helio_time(self):
        period = 0.051451615
        astro_object = get_ztf_example(3)
        astro_object.features.loc[len(astro_object.features)] = (
            "Multiband_period",
            period,
            "g,r",
            "ztf",
            "1.0.0",
        )
        harmonics_extractor = HarmonicsExtractor(
            bands=["g", "r"], unit="magnitude", use_forced_photo=False, n_harmonics=7
        )
        harmonics_extractor.compute_features_single_object(astro_object)
        features = astro_object.features
        mean_chi_original = np.mean(
            features[features["name"] == "Harmonics_chi"]["value"].values
        )

        # plot_astro_object(
        #     astro_object, unit='magnitude',
        #     use_forced_phot=False, period=period)
        lightcurve_preprocessor = ZTFLightcurvePreprocessor()
        lightcurve_preprocessor._helio_time_correction(astro_object)
        # plot_astro_object(
        #     astro_object, unit='magnitude',
        #     use_forced_phot=False, period=period)

        harmonics_extractor.compute_features_single_object(astro_object)
        features = astro_object.features
        mean_chi_helio_time = np.mean(
            features[features["name"] == "Harmonics_chi"]["value"].values
        )

        self.assertLessEqual(mean_chi_helio_time, mean_chi_original)
