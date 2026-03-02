import unittest
from unittest import mock
from apf.producers import GenericProducer
from features.step import FeatureStep
import json
import pandas as pd
import numpy as np
from ..message_factory_lsst import generate_input_batch_lsst

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
            "SURVEY":"lsst"
        }
        db_sql = mock.MagicMock()
        self.step = FeatureStep(config=self.step_config, db_sql=db_sql)
        self.step.scribe_producer = mock.create_autospec(GenericProducer)
        self.step.scribe_producer.produce = mock.MagicMock()

    def test_execute(self):
        messages = generate_input_batch_lsst(50)
        messages = self.step.pre_execute(messages)
        
        result_messages = self.step.execute(messages)

        self.assertEqual(len(messages), len(result_messages))
        n_features_prev = -1
        for result_message in result_messages:
            n_features = len(result_message["features"])
            #self.assertTrue(n_features > 0) #no tengo features por ahora

            # Check all messages have the same number of features
            if n_features_prev != -1:
                self.assertEqual(n_features, n_features_prev)
                n_features_prev = n_features

        scribe_args = self.step.scribe_producer.produce.call_args
        assert (
            len(json.loads(scribe_args[0][0]["payload"])["payload"]["features"])
            == n_features
        )
        self.step.scribe_producer.produce.assert_called()
        scribe_producer_call_count = self.step.scribe_producer.produce.call_count
        self.assertEqual(scribe_producer_call_count, len(messages) * 1)

    def test_post_execute(self):
        messages = generate_input_batch_lsst(10)
        messages = self.step.pre_execute(messages)

        result_messages = self.step.execute(messages)
        result_messages = self.step.post_execute(result_messages)

        for message in result_messages:
            self.assertTrue("reference" not in message.keys())
