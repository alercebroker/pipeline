import unittest
import pandas as pd

from apf.producers import GenericProducer
from xmatch_step import XmatchStep, XmatchClient
from tests.data.messages import generate_input_batch, get_fake_xmatch
from schema import SCHEMA
from unittest import mock

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

XMATCH_CONFIG = {
    "CATALOG": {
        "name": "allwise",
        "columns": [
            "AllWISE",
            "RAJ2000",
            "DEJ2000",
            "W1mag",
            "W2mag",
            "W3mag",
            "W4mag",
            "e_W1mag",
            "e_W2mag",
            "e_W3mag",
            "e_W4mag",
            "Jmag",
            "e_Jmag",
            "Hmag",
            "e_Hmag",
            "Kmag",
            "e_Kmag",
        ],
    }
}


class StepXmatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        step_config = {
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "STEP_METADATA": {
                "STEP_VERSION": "xmatch",
                "STEP_ID": "xmatch",
                "STEP_NAME": "xmatch",
                "STEP_COMMENTS": "xmatch",
            },
            "XMATCH_CONFIG": XMATCH_CONFIG,
            "RETRIES": 3,
            "RETRY_INTERVAL": 1,
        }
        mock_xmatch_client = mock.create_autospec(XmatchClient)
        mock_producer = mock.create_autospec(GenericProducer)
        cls.step = XmatchStep(
            config=step_config,
        )
        cls.batch = generate_input_batch(20)  # I want 20 light  curves

    def test_save_empty_xmatch(self):
        data = pd.DataFrame()
        self.step.save_xmatch(data)
        calls = len(self.step.driver.query().bulk_insert.mock_calls)
        self.assertEqual(calls, 0)

    def test_save_xmatch(self):
        data = get_fake_xmatch(self.batch)
        self.step.save_xmatch(data)
        calls = len(self.step.driver.query().bulk_insert.mock_calls)
        self.assertEqual(calls, 2)  # insert allwise table and xmatch table

    def test_produce(self):
        data = [
            {"oid": ["ATLAS1"], "aid": "A", "non_detections": None},
            {"oid": ["ZTF1"], "aid": "D", "non_detections": None},
            {"oid": ["ZTF2", "ATLAS2"], "aid": "C", "non_detections": None},
            {"oid": ["ATLAS10"], "aid": "D", "non_detections": []},
        ]
        old_calls = len(self.step.producer.produce.mock_calls)
        self.step.produce(data)
        new_calls = len(self.step.producer.produce.mock_calls)
        self.assertEqual(new_calls - old_calls, 4)

    def test_bad_xmatch(self):
        catalog = pd.DataFrame(self.batch)
        catalog.rename(
            columns={"meanra": "ra", "meandec": "dec"}, inplace=True
        )
        self.step.xmatch_client.execute.side_effect = Exception
        with self.assertRaises(Exception) as e:
            self.step.request_xmatch(catalog, retries_count=1)
        self.assertIsInstance(e.exception, Exception)

        with self.assertRaises(Exception) as e:
            self.step.request_xmatch(catalog, retries_count=0)
        self.assertIsInstance(e.exception, Exception)

    @mock.patch("xmatch_step.XmatchStep.save_xmatch")
    def test_execute(self, mock_save_xmatch: mock.Mock):
        old_produce_calls = len(self.step.producer.produce.mock_calls)
        self.step.xmatch_client.execute.return_value = get_fake_xmatch(
            self.batch
        )
        self.step.xmatch_client.execute.side_effect = None
        self.step.execute(self.batch)
        self.step.producer.produce.assert_called()
        mock_save_xmatch.assert_called()
        new_produce_calls = len(self.step.producer.produce.mock_calls)
        self.assertEqual(new_produce_calls - old_produce_calls, 20)
