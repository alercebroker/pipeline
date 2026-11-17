import io

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from hips_dummy_step.metrics import ServiceMetrics

URL = "http://image-service.test/endpoint"

ERROR_DTYPE = np.dtype(
    [("index", "i4"), ("ra", "f4"), ("dec", "f4"), ("reason", "U200")]
)


def npz_payload(n_positions: int, failed: tuple[int, ...] = ()) -> bytes:
    """A response body like the image service's, for n_positions."""
    stamps = np.zeros((n_positions, 1, 5, 30, 30), dtype=np.float32)
    errors = np.array(
        [(i, 0.0, 0.0, "no coverage") for i in failed], dtype=ERROR_DTYPE
    )
    buffer = io.BytesIO()
    np.savez(buffer, stamps=stamps, errors=errors)
    return buffer.getvalue()


@pytest.fixture
def registry() -> CollectorRegistry:
    return CollectorRegistry()


@pytest.fixture
def service_metrics(registry) -> ServiceMetrics:
    return ServiceMetrics(registry=registry)


@pytest.fixture
def sleeps() -> list[float]:
    """Records the waits the client asks for, without waiting."""
    return []
