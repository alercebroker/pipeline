import pytest

from ingestion_step.lsst.strategy import LsstData, LsstStrategy
from ingestion_step.ztf.strategy import ZtfData, ZtfStrategy
from tests.data.generator_lsst import generate_alerts as generate_alerts_lsst
from tests.data.generator_ztf import generate_alerts as generate_alerts_ztf


@pytest.fixture
def ztf_parsed_data() -> ZtfData:
    msgs = list(generate_alerts_ztf())
    parsed_data = ZtfStrategy.parse(msgs)

    return parsed_data


@pytest.fixture
def lsst_parsed_data() -> LsstData:
    msgs = list(generate_alerts_lsst())

    parsed_data = LsstStrategy.parse(msgs)

    return parsed_data
