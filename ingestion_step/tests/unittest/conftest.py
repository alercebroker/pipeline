import pytest

from ingestion_step.core.parser_interface import ParsedData
from ingestion_step.ztf import extractor, parser
from tests.data.generator_ztf import generate_alerts


@pytest.fixture
def ztf_data() -> extractor.ZTFData:
    ztf_data = extractor.extract(list(generate_alerts()))

    return ztf_data


@pytest.fixture
def parsed_ztf_data() -> ParsedData:
    ztf_parser = parser.ZTFParser()

    msgs = list(generate_alerts())
    parsed_ztf_data = ztf_parser.parse(msgs)

    return parsed_ztf_data
