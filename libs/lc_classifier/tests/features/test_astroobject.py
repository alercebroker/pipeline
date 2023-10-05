import pandas as pd
import unittest
from lc_classifier.features.core.base import AstroObject


class TestAstroObject(unittest.TestCase):
    def test_aid_in_metadata(self):
        detections = pd.DataFrame(
            data=[
                [
                    "candid_example",
                    "telescope_atlas1",
                    58206.01,
                    "atlas_survey",
                    "g",
                    "test_program",
                    30.123,
                    0.01,
                    12.15,
                    0.011,
                    15.9,
                    0.3,
                    "magnitude"
                ],
                [
                    "candid_example2",
                    "telescope_atlas1",
                    58207.31,
                    "atlas_survey",
                    "g",
                    "test_program",
                    30.123,
                    0.01,
                    12.15,
                    0.011,
                    15.4,
                    0.2,
                    "magnitude"
                ],
            ],
            columns=[
                "candid",
                "tid",
                "mjd",
                "sid",
                "fid",
                "pid",
                "ra",
                "e_ra",
                "dec",
                "e_dec",
                "brightness",
                "e_brightness",
                "units"
            ]
        )
        metadata = pd.DataFrame(
            [
                ["aid", "aid_example"]
            ],
            columns=["field", "value"]
        )

        astro_object = AstroObject(
            detections=detections,
            metadata=metadata
        )
