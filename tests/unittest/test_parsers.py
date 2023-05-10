import unittest
import pandas as pd
from tests.data.data_for_unittest import messages_for_parsing, features_df_for_parse
from features.utils.parsers import parse_output, parse_scribe_payload

class ParsersTestCase(unittest.TestCase):

    def test_parse_output(self):
        expected_payload = [
            {
                "aid": "aid1",
                "meanra": 888,
                "meandec": 999,
                "detections": [
                ],
                "non_detections": [
                ],
                "xmatches": {
                },
                "features": {
                    "feat1_1": 123,
                    "feat1_2": 456,
                    "feat2": 741,
                    "feat3": 963,
                    "feat4": None
                }
            },
            {
                "aid": "aid2",
                "meanra": 444,
                "meandec": 555,
                "detections": [

                ],
                "non_detections": [

                ],
                "xmatches": {

                },
                "features": {
                    "feat1_1": 321,
                    "feat1_2": 654,
                    "feat2": 147,
                    "feat3": 369,
                    "feat4": 888
                }
            }
        ]
        test_features_df = features_df_for_parse.copy()
        parsed_result = parse_output(test_features_df, messages_for_parsing)

        self.assertEqual(parsed_result, expected_payload)

    def test_parse_scribe_payload(self):
        expected_payload = [
            {
                "collection": "name",
                "type": "update_features",
                "criteria": {"_id": "aid1"},
                "data": {
                    "features_version": "v1",
                    "features": [
                        {
                            "name": "feat1",
                            "fid": 1,
                            "value": 123
                        },
                        {
                            "name": "feat1",
                            "fid": 2,
                            "value": 456
                        },
                        {
                            "name": "feat2",
                            "fid": 12,
                            "value": 741
                        },
                        {
                            "name": "feat3",
                            "fid": None,
                            "value": 963
                        },
                        {
                            "name": "feat4",
                            "fid": None,
                            "value": None
                        }
                    ]
                },
                "options": {"upsert": True}        
            },
            {
                "collection": "name",
                "type": "update_features",
                "criteria": {"_id": "aid2"},
                "data": {
                    "features_version": "v1",
                    "features": [
                        {
                            "name": "feat1",
                            "fid": 1,
                            "value": 321
                        },
                        {
                            "name": "feat1",
                            "fid": 2,
                            "value": 654
                        },
                        {
                            "name": "feat2",
                            "fid": 12,
                            "value": 147
                        },
                        {
                            "name": "feat3",
                            "fid": None,
                            "value": 369
                        },
                        {
                            "name": "feat4",
                            "fid": None,
                            "value": 888
                        }
                    ]
                },
                "options": {"upsert": True}        
            },
        ]
        test_feature_version = "v1"
        test_features_df = features_df_for_parse.copy()
        parsed_result = parse_scribe_payload(
            test_features_df,
            test_feature_version
        )

        self.assertEqual(parsed_result, expected_payload)
