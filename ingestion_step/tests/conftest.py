import random
from typing import Any

import pytest

from tests.data.generator_ztf import generate_alerts

random.seed(42)


@pytest.fixture
def ztf_alerts() -> list[dict[str, Any]]:
    return list(generate_alerts())


# Adds a better assertion message for subset comparison
def pytest_assertrepr_compare(op: str, left: object, right: object):
    if isinstance(left, set) and isinstance(right, set) and op == "<=":
        return [
            "`left` is subset of `right` failed, missing elements from `right`:",
            f"\t{left - right}",
        ]
