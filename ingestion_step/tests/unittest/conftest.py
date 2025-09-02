import random

import pytest

from generator.lsst_alert import LsstAlertGenerator
from ingestion_step.lsst.strategy import LsstData, LsstStrategy
from ingestion_step.ztf.strategy import ZtfData, ZtfStrategy
from tests.data.generator_ztf import generate_alerts as generate_alerts_ztf


@pytest.fixture
def ztf_parsed_data() -> ZtfData:
    msgs = list(generate_alerts_ztf())
    parsed_data = ZtfStrategy.parse(msgs)

    return parsed_data


@pytest.fixture
def lsst_parsed_data() -> LsstData:
    rng = random.Random(42)
    generator = LsstAlertGenerator(rng=rng, new_obj_rate=0.4)
    msgs = [generator.generate_alert() for _ in range(1_000)]

    parsed_data = LsstStrategy.parse(msgs)

    return parsed_data
