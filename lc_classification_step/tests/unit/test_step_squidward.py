import pytest
from tests.unit.test_utils import generate_messages_ztf
from tests.mockdata.features_ztf import features_ztf


@pytest.mark.ztf
def test_step_squidward(test_squidward_model, step_factory_squidward):
    messages = generate_messages_ztf()
    for msg in messages:
        if msg["features"] is None:
            mock_feats = features_ztf()
            msg["features"] = mock_feats
    test_squidward_model(step_factory_squidward, messages)
