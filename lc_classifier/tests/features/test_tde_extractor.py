import unittest

import numpy as np

from lc_classifier.features.extractors.tde_extractor import (
    TDETailExtractor,
    FleetExtractor,
)
from lc_classifier.examples.data import (
    get_ztf_example,
    get_ztf_forced_training_examples,
    get_elasticc_example,
    get_tde_example,
)

# from lc_classifier.utils import plot_astro_object
from lc_classifier.features.extractors.tde_extractor import ColorVariationExtractor


class TestTDETailExtractor(unittest.TestCase):
    def test_no_diff_flux(self):
        astro_object = get_ztf_example(2)
        feature_extractor = TDETailExtractor(bands=list("gr"))
        feature_extractor.compute_features_single_object(astro_object)
        assert np.all(np.isnan(astro_object.features["value"].values))

    def test_ztf_forced_phot(self):
        astro_object = get_tde_example()
        feature_extractor = TDETailExtractor(bands=list("gr"))
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)


class TestFleetExtractor(unittest.TestCase):
    def test_ztf(self):
        # this object has no diff_flux obs
        astro_object = get_ztf_example(2)
        feature_extractor = FleetExtractor(bands=list("gr"))
        feature_extractor.compute_features_single_object(astro_object)
        self.assertTrue(all(np.isnan(astro_object.features["value"])))
        print(astro_object.features)

    def test_ztf_forced_phot_not_transient(self):
        astro_objects = get_ztf_forced_training_examples()
        # plot_astro_object(astro_objects[0], unit='diff_flux', use_forced_phot=True)
        feature_extractor = FleetExtractor(bands=list("gr"))
        astro_object = astro_objects[0]
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_elasticc(self):
        astro_object = get_elasticc_example()
        feature_extractor = FleetExtractor(bands=list("ugrizY"))
        # plot_astro_object(astro_object, unit='diff_flux', use_forced_phot=True)
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)


class TestColorVariationExtractor(unittest.TestCase):
    def test_ztf(self):
        # this object has no diff_flux obs
        astro_object = get_ztf_example(2)
        feature_extractor = ColorVariationExtractor(
            window_len=20, band_1="g", band_2="r"
        )
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_ztf_forced_phot_not_transient(self):
        astro_objects = get_ztf_forced_training_examples()
        astro_object = astro_objects[0]
        feature_extractor = ColorVariationExtractor(
            window_len=20, band_1="g", band_2="r"
        )
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_elasticc(self):
        astro_object = get_elasticc_example()
        feature_extractor = ColorVariationExtractor(
            window_len=20, band_1="i", band_2="z"
        )
        feature_extractor.compute_features_single_object(astro_object)
        self.assertTrue(
            len(astro_object.features.dropna()) == 0
        )  # no obs with 'magnitude', just fluxes
