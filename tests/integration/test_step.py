import unittest
from unittest import mock
import pandas as pd
from apf.producers import GenericProducer
from features.step import FeaturesComputer
from features.core.ztf import ZTFClassifierFeatureExtractor
import pytest
from schema import SCHEMA
from tests.data.message_factory import generate_input_batch, generate_non_ztf_batch
from tests.data.features_factory import generate_features_df

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
    "SCHEMA": SCHEMA,
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "TOPIC": "test-scribe",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": SCHEMA,
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
        self.mock_extractor = mock.MagicMock()
        self.mock_extractor_class = mock.MagicMock()
        self.mock_extractor_class.NAME = "group"
        self.mock_extractor_class.VERSION = "v1"
        self.mock_extractor_class.BANDS_MAPPING = {"g": 1, "r": 2}
        self.mock_extractor_class.return_value = self.mock_extractor
        self.step = FeaturesComputer(
            config=self.step_config,
            features_extractor=self.mock_extractor_class,
        )
        self.step.scribe_producer = mock.create_autospec(GenericProducer)
        self.step.scribe_producer.produce = mock.MagicMock()

    def test_step(self):
        input_messages = generate_input_batch(5)
        input_messages_dataframe = pd.DataFrame(input_messages)

        self.mock_extractor.generate_features.return_value = (
            generate_features_df(input_messages_dataframe)
        )

        result = self.step.execute(input_messages)

        self.assertEqual(len(result), 5)
        for output_message in result:
            self.assertIn("aid", output_message)
            self.assertIn("meanra", output_message)
            self.assertIn("meandec", output_message)
            self.assertIn("detections", output_message)
            self.assertIn("non_detections", output_message)
            self.assertIn("xmatches", output_message)
            self.assertIn("features", output_message)
            self.assertEqual(len(output_message["features"]), 5)
        self.step.scribe_producer.produce.assert_called()
        scribe_producer_call_count = self.step.scribe_producer.produce.call_count
        self.assertEqual(scribe_producer_call_count, 5)
