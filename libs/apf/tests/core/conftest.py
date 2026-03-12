from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from apf.core.step import GenericStep
from apf.metrics.prometheus import PrometheusMetrics


class MockStep(GenericStep):
    def execute(self, _):
        return {}


@pytest.fixture
def registry() -> CollectorRegistry:
    yield CollectorRegistry(auto_describe=True)


@pytest.fixture
def step(registry: CollectorRegistry, config: dict[str, Any]) -> MockStep:
    metrics = PrometheusMetrics(registry=registry)

    step = MockStep(config=config, metrics=metrics)

    yield step
