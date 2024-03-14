import unittest
from unittest import mock
from apf.producers import GenericProducer
from features.step import (
    FeaturesComputer,
)
from tests.data.data_for_unittest import (
    features_df_for_execute,
    messages_for_execute,
)

CONSUMER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "PARAMS": {
        "bootstrap.servers": "server",
        "group.id": "group_id",
        "auto.offset.reset": "beginning",
        "enable.partition.eof": False,
    },
    "TOPICS": ["topic"],
    "consume.messages": "1",
    "consume.timeout": "10",
}

PRODUCER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "TOPIC": "test",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "TOPIC": "test-scribe",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
}


class StepTestCase(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.step_config = {
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
            "FEATURE_VERSION": "v1",
            "STEP_METADATA": {
                "STEP_VERSION": "feature",
                "STEP_ID": "feature",
                "STEP_NAME": "feature",
                "STEP_COMMENTS": "feature",
                "FEATURE_VERSION": "1.0-test",
            },
        }
        self.mock_extractor = mock.MagicMock()
        self.mock_extractor.generate_features.return_value = (
            features_df_for_execute
        )
        self.mock_extractor_class = mock.MagicMock()
        self.mock_extractor_class.NAME = "ztf_lc_features"
        self.mock_extractor_class.VERSION = "v1"
        self.mock_extractor_class.BANDS_MAPPING = {"g": 1, "r": 2}
        self.mock_extractor_class.return_value = self.mock_extractor
        self.step = FeaturesComputer(
            config=self.step_config, extractor=self.mock_extractor_class
        )
        self.step.scribe_producer = mock.create_autospec(GenericProducer)
        self.step.scribe_producer.produce = mock.MagicMock()

    def test_execute(self):
        expected_output = [
            {
                "oid": "oid1",
                "candid": ["1_candid_aid_1", "2_candid_aid_1"],
                "detections": [
                    {
                        "candid": "1_candid_aid_1",
                        "tid": "ztf",
                        "aid": "aid1",
                        "oid": "oid1",
                        "mjd": 111,
                        "sid": "sid-aid1",
                        "fid": "1",
                        "pid": 222,
                        "ra": 333,
                        "e_ra": 444,
                        "dec": 555,
                        "e_dec": 666,
                        "mag": 777,
                        "e_mag": 888,
                        "mag_corr": None,
                        "e_mag_corr": None,
                        "e_mag_corr_ext": None,
                        "isdiffpos": 999,
                        "corrected": False,
                        "dubious": False,
                        "has_stamp": True,
                        "stellar": False,
                        "extra_fields": None,
                    },
                    {
                        "candid": "2_candid_aid_1",
                        "tid": "ztf",
                        "aid": "aid1",
                        "oid": "oid1",
                        "mjd": 111,
                        "sid": "sid-aid1",
                        "fid": "1",
                        "pid": 222,
                        "ra": 333,
                        "e_ra": 444,
                        "dec": 555,
                        "e_dec": 666,
                        "mag": 777,
                        "e_mag": 888,
                        "mag_corr": None,
                        "e_mag_corr": None,
                        "e_mag_corr_ext": None,
                        "isdiffpos": 999,
                        "corrected": False,
                        "dubious": False,
                        "has_stamp": True,
                        "stellar": False,
                        "extra_fields": None,
                    },
                ],
                "non_detections": [
                    {
                        "aid": "aid1",
                        "tid": "ztf",
                        "sid": "sid_aid1",
                        "oid": "oid1",
                        "mjd": 999.888,
                        "fid": "1",
                        "diffmaglim": 123.123,
                    },
                    {
                        "aid": "aid1",
                        "tid": "ztf",
                        "sid": "sid_aid1",
                        "oid": "oid1",
                        "mjd": 888,
                        "fid": "1",
                        "diffmaglim": 999,
                    },
                ],
                "xmatches": {
                    "W1mag": 123,
                    "W2mag": 456,
                    "W3mag": 789,
                },
                "features": {
                    "Amplitude_1": 123,
                    "Amplitude_2": 456,
                    "Multiband_period": 741,
                    "feat3": 963,
                    "rb": None,
                },
            },
            {
                "oid": "oid2",
                "candid": ["1_candid_aid_2"],
                "detections": [
                    {
                        "candid": "1_candid_aid_2",
                        "tid": "ztf",
                        "aid": "aid2",
                        "oid": "oid2",
                        "mjd": 111,
                        "sid": "sid-aid1",
                        "fid": "1",
                        "pid": 222,
                        "ra": 333,
                        "e_ra": 444,
                        "dec": 555,
                        "e_dec": 666,
                        "mag": 777,
                        "e_mag": 888,
                        "mag_corr": None,
                        "e_mag_corr": None,
                        "e_mag_corr_ext": None,
                        "isdiffpos": 999,
                        "corrected": False,
                        "dubious": False,
                        "has_stamp": True,
                        "stellar": False,
                        "extra_fields": None,
                    }
                ],
                "non_detections": [
                    {
                        "aid": "aid2",
                        "tid": "ztf",
                        "sid": "sid_aid2",
                        "oid": "oid2",
                        "mjd": 888,
                        "fid": "1",
                        "diffmaglim": 999,
                    }
                ],
                "xmatches": {
                    "W1mag": 123,
                    "W2mag": 456,
                    "W3mag": 789,
                },
                "features": {
                    "Amplitude_1": 321,
                    "Amplitude_2": 654,
                    "Multiband_period": 147,
                    "feat3": 369,
                    "rb": 888,
                },
            },
        ]
        expected_detections_for_extractor = [
            {
                "candid": "1_candid_aid_1",
                "tid": "ztf",
                "aid": "aid1",
                "oid": "oid1",
                "mjd": 111,
                "sid": "sid-aid1",
                "fid": "1",
                "pid": 222,
                "ra": 333,
                "e_ra": 444,
                "dec": 555,
                "e_dec": 666,
                "mag": 777,
                "e_mag": 888,
                "mag_corr": None,
                "e_mag_corr": None,
                "e_mag_corr_ext": None,
                "isdiffpos": 999,
                "corrected": False,
                "dubious": False,
                "has_stamp": True,
                "stellar": False,
                "extra_fields": None,
                "index_column": "1_candid_aid_1_oid1",
            },
            {
                "candid": "2_candid_aid_1",
                "tid": "ztf",
                "aid": "aid1",
                "oid": "oid1",
                "mjd": 111,
                "sid": "sid-aid1",
                "fid": "1",
                "pid": 222,
                "ra": 333,
                "e_ra": 444,
                "dec": 555,
                "e_dec": 666,
                "mag": 777,
                "e_mag": 888,
                "mag_corr": None,
                "e_mag_corr": None,
                "e_mag_corr_ext": None,
                "isdiffpos": 999,
                "corrected": False,
                "dubious": False,
                "has_stamp": True,
                "stellar": False,
                "extra_fields": None,
                "index_column": "2_candid_aid_1_oid1",
            },
            {
                "candid": "1_candid_aid_2",
                "tid": "ztf",
                "aid": "aid2",
                "oid": "oid2",
                "mjd": 111,
                "sid": "sid-aid1",
                "fid": "1",
                "pid": 222,
                "ra": 333,
                "e_ra": 444,
                "dec": 555,
                "e_dec": 666,
                "mag": 777,
                "e_mag": 888,
                "mag_corr": None,
                "e_mag_corr": None,
                "e_mag_corr_ext": None,
                "isdiffpos": 999,
                "corrected": False,
                "dubious": False,
                "has_stamp": True,
                "stellar": False,
                "extra_fields": None,
                "index_column": "1_candid_aid_2_oid2",
            },
        ]
        expected_non_detections_for_extractor = [
            {
                "aid": "aid1",
                "tid": "ztf",
                "sid": "sid_aid1",
                "oid": "oid1",
                "mjd": 999.888,
                "fid": "1",
                "diffmaglim": 123.123,
            },
            {
                "aid": "aid1",
                "tid": "ztf",
                "sid": "sid_aid1",
                "oid": "oid1",
                "mjd": 888,
                "fid": "1",
                "diffmaglim": 999,
            },
            {
                "aid": "aid2",
                "tid": "ztf",
                "sid": "sid_aid2",
                "oid": "oid2",
                "mjd": 888,
                "fid": "1",
                "diffmaglim": 999,
            },
        ]
        expected_xmatches_for_extractor = [
            {
                "oid": "oid1",
                "W1mag": 123,
                "W2mag": 456,
                "W3mag": 789,
            },
            {
                "oid": "oid2",
                "W1mag": 123,
                "W2mag": 456,
                "W3mag": 789,
            },
        ]

        result = self.step.execute(messages_for_execute)

        self.mock_extractor_class.assert_called_once_with(
            expected_detections_for_extractor,
            expected_non_detections_for_extractor,
            expected_xmatches_for_extractor,
        )
        self.assertEqual(result, expected_output)

        self.step.scribe_producer.produce.assert_called()
        scribe_producer_call_count = (
            self.step.scribe_producer.produce.call_count
        )
        self.assertEqual(scribe_producer_call_count, 2)
