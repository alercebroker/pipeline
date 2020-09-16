import unittest
from unittest import mock
from s3_step.step import upload_file
from s3_step.step import S3Step


class StepTestCase(unittest.TestCase):
    def setUp(self):
        STORAGE_CONFIG = {"BUCKET_NAME": "fake_bucket"}
        STEP_METADATA = {
            "STEP_VERSION": "dev",
            "STEP_ID": "s3",
            "STEP_NAME": "s3",
            "STEP_COMMENTS": "s3 upload",
        }
        METRICS_CONFIG = {}
        self.step_config = {
            "STORAGE": STORAGE_CONFIG,
            "STEP_METADATA": STEP_METADATA,
            "METRICS_CONFIG": METRICS_CONFIG,
        }
        self.step = S3Step(
            config=self.step_config,
        )

    def test_jd_to_date(self):
        jd = 2459110.5
        date = self.step.jd_to_date(jd)
        real = (2020, 9, 18)
        self.assertEqual(real, date)

