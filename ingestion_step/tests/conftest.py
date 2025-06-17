import random

import pytest

import tests.data.generator_lsst as lsst
import tests.data.generator_ztf as ztf
from ingestion_step.core.types import Message

random.seed(42)


@pytest.fixture
def ztf_alerts() -> list[Message]:
    return list(ztf.generate_alerts())


@pytest.fixture
def lsst_alerts() -> list[Message]:
    return list(lsst.generate_alerts())


# Adds a better assertion message for subset comparison
def pytest_assertrepr_compare(op: str, left: object, right: object):
    if isinstance(left, set) and isinstance(right, set) and op == "<=":
        return [
            "`left` is subset of `right` failed, missing elements from `right`:",
            f"\t{left - right}",
        ]
