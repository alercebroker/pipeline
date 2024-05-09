import unittest
from lc_classifier.features.extractors.allwise_colors_feature_extractor import (
    AllwiseColorsFeatureExtractor,
)
from lc_classifier.examples.data import get_ztf_forced_phot_cepheid


class TestAllWiseFeatureExtractor(unittest.TestCase):
    def test_ztf_forced_phot(self):
        astro_object = get_ztf_forced_phot_cepheid()

        bands = ["g", "r"]
        feature_extractor = AllwiseColorsFeatureExtractor(bands)
        feature_extractor.compute_features_single_object(astro_object)
        features = astro_object.features

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
        feature_extractor = AllwiseColorsFeatureExtractor(bands)
        feature_extractor.compute_features_single_object(astro_object)
        features = astro_object.features
