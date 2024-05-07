import unittest
from lc_classifier.features.extractors.period_extractor import PeriodExtractor
from lc_classifier.examples.data import get_ztf_example
from lc_classifier.examples.data import get_elasticc_example_2
from lc_classifier.examples.data import get_ztf_forced_phot_cepheid
import numpy as np
import pandas as pd


class TestPeriodFeatureExtractor(unittest.TestCase):
    def test_ztf_object(self):
        ztf_astro_object = get_ztf_example(1)
        feature_extractor = PeriodExtractor(
            bands=["g", "r"],
            unit="magnitude",
            smallest_period=0.045,
            largest_period=500.0,
            trim_lightcurve_to_n_days=1500.0,
            min_length=15,
            use_forced_photo=False,
            return_power_rates=False,
        )
        feature_extractor.compute_features_single_object(ztf_astro_object)
        period = ztf_astro_object.features[
            ztf_astro_object.features["name"] == "Multiband_period"
        ]
        period = period["value"].values[0]
        self.assertLessEqual(np.abs(0.5388 - period), 0.01)

    def test_trim_with_elasticc(self):
        elasticc_astro_object = get_elasticc_example_2()
        feature_extractor = PeriodExtractor(
            bands=list("ugrizY"),
            unit="diff_flux",
            smallest_period=0.045,
            largest_period=50.0,
            trim_lightcurve_to_n_days=500.0,
            min_length=15,
            use_forced_photo=True,
            return_power_rates=False,
        )
        all_obs = pd.concat(
            [elasticc_astro_object.detections, elasticc_astro_object.forced_photometry],
            axis=0,
        )
        original_timespan = all_obs["mjd"].max() - all_obs["mjd"].min()
        self.assertGreater(original_timespan, 500.0)
        trimmed_lc = feature_extractor.get_observations(elasticc_astro_object)
        trimmed_lc_timespan = trimmed_lc["mjd"].max() - trimmed_lc["mjd"].min()
        self.assertLessEqual(trimmed_lc_timespan, 500.0)

    def test_power_rates(self):
        ztf_astro_object = get_ztf_example(1)
        feature_extractor = PeriodExtractor(
            bands=["g", "r"],
            unit="magnitude",
            smallest_period=0.045,
            largest_period=500.0,
            trim_lightcurve_to_n_days=1500.0,
            min_length=15,
            use_forced_photo=False,
            return_power_rates=True,
        )
        feature_extractor.compute_features_single_object(ztf_astro_object)
        features = ztf_astro_object.features
        pr = features[
            features["name"].isin(
                [f for f in features["name"].values if "Power_rate" in f]
            )
        ]
        self.assertLessEqual(np.max(pr["value"]), 0.1)

    def test_lc_too_short(self):
        ztf_astro_object = get_ztf_example(1)
        bands = ["g", "r"]
        feature_extractor = PeriodExtractor(
            bands=bands,
            unit="magnitude",
            smallest_period=0.045,
            largest_period=500.0,
            trim_lightcurve_to_n_days=1500.0,
            min_length=1_000,
            use_forced_photo=False,
            return_power_rates=True,
        )
        feature_extractor.compute_features_single_object(ztf_astro_object)
        features = ztf_astro_object.features
        self.assertEqual(
            len(features), len(feature_extractor.factors) + 2 + 2 * len(bands)
        )

    def test_lc_single_band_obs(self):
        astro_object = get_ztf_forced_phot_cepheid()
        selected_band = "g"
        astro_object.detections = astro_object.detections[
            astro_object.detections["fid"] == selected_band
        ]
        astro_object.forced_photometry = astro_object.forced_photometry[
            astro_object.forced_photometry["fid"] == selected_band
        ]

        bands = ["g", "r"]
        feature_extractor = PeriodExtractor(
            bands=bands,
            unit="magnitude",
            smallest_period=0.045,
            largest_period=500.0,
            trim_lightcurve_to_n_days=500.0,
            min_length=15,
            use_forced_photo=True,
            return_power_rates=True,
        )
        feature_extractor.compute_features_single_object(astro_object)
        features = astro_object.features
        print(features)

    def test_lc_band_with_few_obs(self):
        astro_object = get_ztf_forced_phot_cepheid()

        bands = ["g", "r"]
        feature_extractor = PeriodExtractor(
            bands=bands,
            unit="magnitude",
            smallest_period=0.045,
            largest_period=500.0,
            trim_lightcurve_to_n_days=500.0,
            min_length=15,
            use_forced_photo=False,
            return_power_rates=True,
        )
        feature_extractor.compute_features_single_object(astro_object)
        features = astro_object.features
        print(features)
