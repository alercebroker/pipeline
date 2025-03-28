import pytest

from ingestion_step.ztf import extractor
from tests.data.generator_ztf import generate_alerts


@pytest.fixture
def ztf_data() -> extractor.ZTFData:
    ztf_data = extractor.extract(list(generate_alerts()))

    return ztf_data
