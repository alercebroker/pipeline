import numpy as np
import pandas as pd
import unittest

from lc_classifier.features.extractors.coordinate_extractor import CoordinateExtractor
from lc_classifier.examples.data import get_ztf_example, get_elasticc_example
from lc_classifier.features.core.base import AstroObject


class TestCoordinateExtractor(unittest.TestCase):
    def test_ztf_object(self):
        astro_object = get_ztf_example(2)
        feature_extractor = CoordinateExtractor()
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_elasticc_object(self):
        astro_object = get_elasticc_example()
        feature_extractor = CoordinateExtractor()
        feature_extractor.compute_features_single_object(astro_object)
        print(astro_object.features)

    def test_north_pole(self):
        metadata = pd.DataFrame(
            data=(("aid", "aid_example"),), columns=["name", "value"]
        )
        detections = pd.DataFrame(
            data=(
                (
                    120.0,
                    90.0,
                    "sid_example",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ),
            columns=[
                "ra",
                "dec",
                "sid",
                "candid",
                "tid",
                "mjd",
                "fid",
                "pid",
                "brightness",
                "e_brightness",
                "unit",
            ],
        )
        astro_object = AstroObject(metadata=metadata, detections=detections)
        feature_extractor = CoordinateExtractor()
        feature_extractor.compute_features_single_object(astro_object)
        coordinates = astro_object.features["value"].values.flatten()
        self.assertLessEqual(np.abs(coordinates[0]), 1e-6)
        self.assertLessEqual(np.abs(coordinates[1]), 1e-6)
        self.assertGreaterEqual(np.abs(coordinates[2]), 1 - 1e-6)
