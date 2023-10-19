import unittest
from unittest import mock
import pandas as pd
from tests.data.data_for_unittest import (
    messages_for_parsing,
    features_df_for_parse,
)
from features.utils.parsers import parse_output, parse_scribe_payload
from importlib import metadata


class ParsersTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_extractor_class = mock.MagicMock
        self.mock_extractor_class.NAME = "ztf_lc_features"
        self.mock_extractor_class.VERSION = metadata.version("feature-step")
        self.mock_extractor_class.BANDS_MAPPING = {"g": 1, "r": 2}

    def test_parse_output(self):
        expected_payload = [
            {
                "aid": "aid1",
                "meanra": 888,
                "meandec": 999,
                "detections": [],
                "non_detections": [],
                "xmatches": {},
                "features": {
                    "feat1_1": 123,
                    "feat1_2": 456,
                    "feat2": 741,
                    "feat3": 963,
                    "feat4": None,
                },
            },
            {
                "aid": "aid2",
                "meanra": 444,
                "meandec": 555,
                "detections": [],
                "non_detections": [],
                "xmatches": {},
                "features": {
                    "feat1_1": 321,
                    "feat1_2": 654,
                    "feat2": 147,
                    "feat3": 369,
                    "feat4": 888,
                },
            },
        ]
        test_features_df = features_df_for_parse.copy()
        parsed_result = parse_output(
            test_features_df, messages_for_parsing, self.mock_extractor_class
        )

        self.assertEqual(parsed_result, expected_payload)

    def test_parse_scribe_payload(self):
        expected_payload = [
            {
                "collection": "object",
                "type": "update_features",
                "criteria": {"_id": "aid1", "oid": ["oid1", "oid6", "oid7"]},
                "data": {
                    "features_version": metadata.version("feature-step"),
                    "features_group": "ztf_lc_features",
                    "features": [
                        {"name": "feat1", "fid": 1, "value": 123},
                        {"name": "feat1", "fid": 2, "value": 456},
                        {"name": "feat2", "fid": 12, "value": 741},
                        {"name": "feat3", "fid": 0, "value": 963},
                        {"name": "feat4", "fid": 0, "value": None},
                    ],
                },
                "options": {"upsert": True},
            },
            {
                "collection": "object",
                "type": "update_features",
                "criteria": {"_id": "aid2", "oid": ["oid2", "oid5", "oid8"]},
                "data": {
                    "features_version": metadata.version("feature-step"),
                    "features_group": "ztf_lc_features",
                    "features": [
                        {"name": "feat1", "fid": 1, "value": 321},
                        {"name": "feat1", "fid": 2, "value": 654},
                        {"name": "feat2", "fid": 12, "value": 147},
                        {"name": "feat3", "fid": 0, "value": 369},
                        {"name": "feat4", "fid": 0, "value": 888},
                    ],
                },
                "options": {"upsert": True},
            },
        ]
        test_features_df = features_df_for_parse.copy()
        messages_aid_oid = {
            "aid1": ["oid1", "oid6", "oid7"],
            "aid2": ["oid2", "oid5", "oid8"],
            "aid3": ["oid3", "oid4", "oid9"],
        }
        parsed_result = parse_scribe_payload(
            messages_aid_oid, test_features_df, self.mock_extractor_class
        )

        self.assertEqual(parsed_result, expected_payload)
