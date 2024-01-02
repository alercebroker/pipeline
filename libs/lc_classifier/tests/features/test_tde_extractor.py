import unittest

import numpy as np

from lc_classifier.features.extractors.tde_extractor import TDETailExtractor, FleetExtractor
from lc_classifier.examples.data import get_ztf_example, get_ztf_forced_training_examples, get_elasticc_example
from lc_classifier.utils import plot_astro_object


class TestTDETailExtractor(unittest.TestCase):
    def test_ztf(self):
        astro_object = get_ztf_example(2)
        feature_extractor = TDETailExtractor(
            bands=list('gr'))
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_ztf_forced_phot(self):
        astro_objects = get_ztf_forced_training_examples()
        feature_extractor = TDETailExtractor(
            bands=list('gr'))
        astro_object = astro_objects[0]
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)


class TestFleetExtractor(unittest.TestCase):
    def test_ztf(self):
        # this object has no diff_flux obs
        astro_object = get_ztf_example(2)
        feature_extractor = FleetExtractor(
            bands=list('gr'))
        feature_extractor.compute_features_single_object(astro_object)
        self.assertTrue(all(np.isnan(astro_object.features['value'])))
        print(astro_object.features)

    def test_ztf_forced_phot_not_transient(self):
        astro_objects = get_ztf_forced_training_examples()
        feature_extractor = FleetExtractor(
            bands=list('gr'))
        astro_object = astro_objects[0]
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_elasticc(self):
        astro_object = get_elasticc_example()
        feature_extractor = FleetExtractor(
            bands=list('ugrizY'))
        # plot_astro_object(astro_object, unit='diff_flux', use_forced_phot=True)
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)
