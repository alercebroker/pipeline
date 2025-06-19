from itertools import count
from unittest.mock import MagicMock

from ingestion_step.core.types import Message
from ingestion_step.lsst.strategy import LsstStrategy


def test_parse(lsst_alerts: list[Message], mock_driver: MagicMock):
    # Only checking that it does not crash
    # TODO: Implement proper test

    LsstStrategy.parse(lsst_alerts, mock_driver)
