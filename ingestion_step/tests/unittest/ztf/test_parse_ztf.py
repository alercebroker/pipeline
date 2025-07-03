from ingestion_step.core.types import Message
from ingestion_step.ztf.strategy import ZtfStrategy


def test_parse(ztf_alerts: list[Message]):
    # Only checking that it does not crash
    # TODO: Implement proper test
    ZtfStrategy.parse(ztf_alerts)
