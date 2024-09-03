import pytest
from tests.unit.test_utils import generate_messages_elasticc


@pytest.mark.elasticc
def test_step_mlp(test_elasticc_model, step_factory_mlp):
    test_elasticc_model(step_factory_mlp, generate_messages_elasticc())
