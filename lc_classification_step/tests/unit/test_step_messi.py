import pytest
from tests.unit.test_utils import generate_messages_elasticc


@pytest.mark.elasticc
def test_step_messi(test_elasticc_model, step_factory_messi):
    test_elasticc_model(step_factory_messi, generate_messages_elasticc())
