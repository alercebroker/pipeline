import pathlib
import unittest
from unittest import mock

from apf.producers import GenericProducer
from tests.data.messages import (
    generate_input_batch,
    generate_non_ztf_batch,
    get_fake_empty_xmatch,
    get_fake_xmatch,
)
from xmatch_step import XmatchClient, XmatchStep

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
    "SCHEMA_PATH": pathlib.Path(
        pathlib.Path(__file__).parent.parent.parent.parent,
        "schemas/xmatch_step",
        "output.avsc",
    ),
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "TOPIC": "test",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA": pathlib.Path(
        pathlib.Path(__file__).parent.parent.parent.parent,
        "schemas/scribe_step",
        "scribe.avsc",
    ),
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

        def mock_method(a):
            return None

        cls.step.producer.set_key_field = mock_method
        cls.batch = generate_input_batch(20)  # I want 20 light  curves

    @mock.patch.object(XmatchClient, "execute")
    def test_execute(self, xmatch_client):
        self.step.producer.set_key_field = mock.MagicMock()
        xmatch_client.return_value = get_fake_xmatch(self.batch)
        self.step.scribe_producer = mock.MagicMock()
        xmatches, lightcurves_by_oid, candids = self.step.execute(self.batch)
        xmatches, lightcurves_by_oid, candids = self.step.post_execute(
            (xmatches, lightcurves_by_oid, candids)
        )
        output = self.step.pre_produce((xmatches, lightcurves_by_oid, candids))
        assert isinstance(output, list)
        assert len(output) == 20
        for obj in output:
            assert "xmatches" in obj
            assert "allwise" in obj["xmatches"]
            assert "candid" in obj
            assert obj["candid"] is not None
            assert obj["xmatches"]["allwise"] is not None

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

    @mock.patch.object(XmatchClient, "execute")
    def test_execute_empty_xmatch(self, mock_xmatch: mock.Mock):
        self.step.producer.set_key_field = mock.MagicMock()
        self.step.scribe_producer = mock.MagicMock()
        mock_xmatch.return_value = get_fake_empty_xmatch(self.batch)
        xmatches, oids, candids = self.step.execute(self.batch)
        for oid in oids.values():
            assert isinstance(oid, dict)
        assert len(oids) == 20
        assert xmatches.shape == (0, 5)
