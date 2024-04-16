from json import loads
from unittest.mock import MagicMock

import pytest
from lc_classification.core.step import LateClassifier
from tests.test_commons import (
    assert_command_is_correct,
    assert_ztf_object_is_correct,
)
from tests.unit.test_utils import generate_messages_ztf


@pytest.mark.ztf
def test_step(step_factory_ztf):
    step: LateClassifier = step_factory_ztf(generate_messages_ztf())
    step.start()
    scribe_calls = step.scribe_producer.mock_calls
    # Tests scribe produces correct commands
    assert len(scribe_calls) > 0
    for call in scribe_calls:
        message = loads(call.args[0]["payload"])
        assert_command_is_correct(message)
    # Test producer produces correct data
    assert isinstance(step.producer, MagicMock)
    calls = step.producer.mock_calls
    assert len(calls) > 0
    for call in calls:
        if len(call.args) > 1: # because of __del__
            obj = call.args[0]
            assert_ztf_object_is_correct(obj)


@pytest.mark.ztf
def test_step_empty_features(step_factory_ztf):
    from lc_classifier.classifier.models import HierarchicalRandomForest

    model = HierarchicalRandomForest()
    model.download_model()
    model.load_model(model.MODEL_PICKLE_PATH)
    empty_features = []
    for msg in generate_messages_ztf():
        msg.pop("features")
        empty_features.append(msg)
    step = step_factory_ztf(empty_features)
    step.model = model
    step.start()
    scribe_calls = step.scribe_producer.mock_calls
    assert scribe_calls == []
    calls = step.producer.mock_calls
    assert len(calls) == 1 # because of __del__
