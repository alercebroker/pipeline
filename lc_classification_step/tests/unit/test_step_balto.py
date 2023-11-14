import pytest
from tests.unit.test_utils import generate_messages_elasticc


@pytest.mark.elasticc
def test_step_balto(test_elasticc_model, step_factory_balto):
    test_elasticc_model(step_factory_balto, generate_messages_elasticc())
