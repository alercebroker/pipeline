import random
from typing import Any

import numpy as np
import pytest

from ingestion_step.ztf import extractor
from tests.unittest.data.generator_ztf import generate_alerts

random.seed(42)


@pytest.fixture
def ztf_alerts() -> list[dict[str, Any]]:
    return list(generate_alerts())


@pytest.fixture
def ztf_data() -> extractor.ZTFData:
    ztf_data = extractor.extract(list(generate_alerts()))

    # generate_alerts is totally random
    # Some fields need to be re define to valid ranges
    ztf_data["candidates"]["fid"] = np.random.randint(
        1, 4, size=len(ztf_data["candidates"])
    )
    ztf_data["prv_candidates"]["fid"] = np.random.randint(
        1, 4, size=len(ztf_data["prv_candidates"])
    )
    ztf_data["fp_hist"]["fid"] = np.random.randint(1, 4, size=len(ztf_data["fp_hist"]))

    return ztf_data
