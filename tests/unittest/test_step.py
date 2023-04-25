import unittest
import pandas as pd

from apf.producers import GenericProducer
from xmatch_step import XmatchStep, XmatchClient
from tests.data.messages import (
    generate_input_batch,
    get_fake_xmatch,
    generate_non_ztf_batch,
)
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

SCRIBE_PRODUCER_CONFIG = {
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
            "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
            "XMATCH_CONFIG": XMATCH_CONFIG,
            "RETRIES": 3,
            "RETRY_INTERVAL": 1,
        }

        cls.step = XmatchStep(
            config=step_config,
        )
        cls.batch = generate_input_batch(20)  # I want 20 light  curves

    @mock.patch.object(XmatchClient, "execute")
    def test_execute(self, xmatch_client):
        xmatch_client.return_value = get_fake_xmatch(self.batch)
        self.step.scribe_producer = mock.MagicMock()
        result = self.step.execute(self.batch)
        assert isinstance(result, tuple)
        output = self.step.post_execute(result)
        messages_with_nd = self.step.pre_produce(output)
        assert isinstance(messages_with_nd, list)

    # Just for coverage (btw, now accepts non ztf objects)
    @mock.patch.object(XmatchClient, "execute")
    def test_non_ztf_only_messages(self, xmatch_client):
        non_ztf_batch = generate_non_ztf_batch(10)
        xmatch_client.return_value = get_fake_xmatch(non_ztf_batch)
        result = self.step.execute(non_ztf_batch)
        assert isinstance(result, tuple)
        assert len(result[0]) == 10

        self.step.scribe_producer = mock.create_autospec(GenericProducer)
        self.step.scribe_producer.produce = mock.MagicMock()

        self.step.post_execute(result)
        self.step.scribe_producer.produce.assert_called()
