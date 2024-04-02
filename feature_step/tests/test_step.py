import unittest
from unittest import mock
from apf.producers import GenericProducer
from features.step import FeatureStep
from .message_factory import generate_input_batch


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
        self.step = FeatureStep(config=self.step_config)
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
                        "extra_fields": {},
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
                        "extra_fields": {},
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
                        "extra_fields": {},
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

        messages = generate_input_batch(
            10,
            ["g", "r"],
            survey="ZTF")
        result_messages = self.step.execute(messages)

        self.assertEqual(len(messages), len(result_messages))
        for result_message in result_messages:
            self.assertTrue(len(result_message['features']) > 0)

        self.step.scribe_producer.produce.assert_called()
        scribe_producer_call_count = (
            self.step.scribe_producer.produce.call_count
        )
        self.assertEqual(scribe_producer_call_count, len(messages))
