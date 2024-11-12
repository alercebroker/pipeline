import unittest
from unittest import mock
from apf.producers import GenericProducer
from features.step import FeatureStep
import json
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from ..message_factory import generate_input_batch
from .message_example import messages as spm_messages
from features.utils.parsers import extract_reference


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
        db_sql = mock.MagicMock()
        self.step = FeatureStep(config=self.step_config, db_sql=db_sql)
        self.step.scribe_producer = mock.create_autospec(GenericProducer)
        self.step.scribe_producer.produce = mock.MagicMock()

    def test_execute(self):
        messages = generate_input_batch(10, ["g", "r"], survey="ZTF")
        result_messages = self.step.execute(messages)

        self.assertEqual(len(messages), len(result_messages))
        n_features_prev = -1
        for result_message in result_messages:
            n_features = len(result_message["features"])
            self.assertTrue(n_features > 0)

            # Check all messages have the same number of features
            if n_features_prev != -1:
                self.assertEqual(n_features, n_features_prev)
                n_features_prev = n_features

        scribe_args = self.step.scribe_producer.produce.call_args
        assert (
            len(json.loads(scribe_args[0][0]["payload"])["data"]["features"])
            == n_features
        )
        self.step.scribe_producer.produce.assert_called()
        scribe_producer_call_count = self.step.scribe_producer.produce.call_count
        # 2 times the len of messages 1 for objects and 1 for features
        self.assertEqual(scribe_producer_call_count, len(messages) * 2)

    def test_tough_examples(self):
        messages = spm_messages
        result_messages = self.step.execute(messages)

        self.assertEqual(len(messages), len(result_messages))
        n_features_prev = -1
        for result_message in result_messages:
            n_features = len(result_message["features"])
            self.assertTrue(n_features > 0)

            # Check all messages have the same number of features
            if n_features_prev != -1:
                self.assertEqual(n_features, n_features_prev)
                n_features_prev = n_features

        self.step.scribe_producer.produce.assert_called()
        scribe_producer_call_count = self.step.scribe_producer.produce.call_count

    def test_period_consistency(self):
        messages = generate_input_batch(10, ["g", "r"], survey="ZTF")
        result_messages = self.step.execute(messages)

        self.assertEqual(len(messages), len(result_messages))
        result_message = result_messages[-1]
        scribe_args = self.step.scribe_producer.produce.call_args
        features = json.loads(scribe_args[0][0]["payload"])["data"]["features"]
        features = pd.DataFrame.from_records(features)
        multiband_period_scribe = features[features["name"] == "Multiband_period"][
            "value"
        ].values.astype(np.float64)[0]
        multiband_period_message = result_message["features"]["Multiband_period_12"]

        assert multiband_period_scribe == multiband_period_message

    @mock.patch("features.step._get_sql_references")
    def test_read_empty_reference_from_db(self, _get_sql_ref):
        _get_sql_ref.return_value = []

        messages = [spm_messages[1]]
        oids = set()
        for msg in messages:
            oids.add(msg["oid"])
        result_references_from_db = self.step.get_sql_references(oids)

        self.assertEqual(0, len(result_references_from_db))

    @mock.patch("features.step._get_sql_references")
    def test_references_are_in_messages_only(self, _get_sql_ref):
        _get_sql_ref.return_value = []

        messages = [spm_messages[2]]
        result_messages = self.step.execute(messages)

        self.assertEqual(2, len(result_messages[0]["reference"]))

    @mock.patch("features.step._get_sql_references")
    def test_references_some_are_in_db(self, _get_sql_ref):
        rfid = 783120150
        _get_sql_ref.return_value = [
            {
                "oid": "ZTF18acphgdi",
                "candid": "5",
                "rfid": rfid,
                "sharpnr": -0.020999999716877937,
                "chinr": 0.5440000295639038,
                "fid": "g",
            },
        ]

        columns = ["oid", "rfid", "sharpnr", "chinr"]

        messages = [spm_messages[3]]
        result_messages = self.step.execute(messages)

        result_references = pd.DataFrame(result_messages[0]["reference"]).set_index(
            "rfid"
        )
        result_references_from_db = pd.DataFrame(_get_sql_ref.return_value)[
            columns
        ].set_index("rfid")

        assert_frame_equal(
            pd.DataFrame(result_references.loc[rfid]),
            pd.DataFrame(result_references_from_db.loc[rfid]),
            check_like=True,
        )

    @mock.patch("features.step._get_sql_references")
    def test_references_are_all_in_db(self, _get_sql_ref):
        _get_sql_ref.return_value = [
            {
                "oid": "ZTF23abuesxr",
                "candid": "12",
                "rfid": 783120108,
                "sharpnr": -0.024000000208616257,
                "chinr": 0.5049999952316284,
                "fid": "g",
            },
            {
                "oid": "ZTF23abuesxr",
                "candid": "17",
                "rfid": 783120208,
                "sharpnr": -0.08399999886751175,
                "chinr": 1.0240000486373901,
                "fid": "r",
            },
        ]

        columns = ["oid", "rfid", "sharpnr", "chinr"]

        messages = [spm_messages[0]]
        result_messages = self.step.execute(messages)
        # print(result_messages[0]["reference"])

        assert_frame_equal(
            pd.DataFrame(result_messages[0]["reference"]).set_index("rfid"),
            pd.DataFrame(_get_sql_ref.return_value)[columns].set_index("rfid"),
            check_like=True,
        )

    def test_post_execute(self):
        messages = generate_input_batch(10, ["g", "r"], survey="ZTF")
        result_messages = self.step.execute(messages)
        result_messages = self.step.post_execute(result_messages)

        for message in result_messages:
            self.assertTrue("reference" not in message.keys())
