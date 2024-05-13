import pandas as pd
import unittest
from lc_classifier.features.core.base import AstroObject


class TestAstroObject(unittest.TestCase):
    def setUp(self) -> None:
        self.detections = pd.DataFrame(
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
                    "magnitude",
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
                    "magnitude",
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
                "unit",
            ],
        )

        self.metadata = pd.DataFrame(
            [["aid", "aid_example"]], columns=["name", "value"]
        )

    def test_aid_in_metadata(self):
        astro_object = AstroObject(detections=self.detections, metadata=self.metadata)

        incomplete_metadata = pd.DataFrame(
            [["oid", "oid_example"]], columns=["name", "value"]
        )

        with self.assertRaises(ValueError) as cm:
            astro_object = AstroObject(
                detections=self.detections, metadata=incomplete_metadata
            )

    def test_detections_missing_columns(self):
        with self.assertRaises(ValueError) as cm:
            astro_object = AstroObject(
                detections=self.detections[["candid", "mjd"]], metadata=self.metadata
            )
