import pytest
from tests.unit.test_utils import (
    generate_messages_ztf,
)


@pytest.mark.ztf
def test_step_anomaly(test_anomaly_model, step_factory_anomaly):
    test_anomaly_model(step_factory_anomaly, generate_messages_ztf())
