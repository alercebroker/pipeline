import unittest

from lc_classifier.features.extractors.spm_extractor import SPMExtractor
from lc_classifier.examples.data import (
    get_elasticc_example,
    get_ztf_forced_training_examples,
)


class TestSPMExtractor(unittest.TestCase):
    def test_with_elasticc(self):
        astro_object = get_elasticc_example()
        feature_extractor = SPMExtractor(
            bands=list("ugrizY"),
            unit="diff_flux",
            redshift="REDSHIFT_HELIO",
            extinction_color_excess="MWEBV",
            forced_phot_prelude=30.0,
        )
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_ztf_forced_phot(self):
        astro_objects = get_ztf_forced_training_examples()
        feature_extractor = SPMExtractor(
            bands=list("gr"),
            unit="diff_flux",
            redshift=None,
            extinction_color_excess=None,
            forced_phot_prelude=30.0,
        )
        astro_object = astro_objects[0]
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)
