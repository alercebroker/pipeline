from json import loads

import pytest
from tests.test_commons import (
    assert_command_is_correct,
    assert_elasticc_object_is_correct,
)
from tests.unit.test_utils import generate_messages_elasticc


@pytest.mark.elasticc
def test_step_toretto(test_elasticc_model, step_factory_toretto):
    test_elasticc_model(step_factory_toretto, generate_messages_elasticc())


@pytest.mark.elasticc
def test_step_toretto_model_input_is_correct(step_factory_toretto):
    step = step_factory_toretto(generate_messages_elasticc())
    step.start()
    predictor_calls = step.model.predict.mock_calls
    assert len(predictor_calls) > 0
    for call in predictor_calls:
        # check that there are features in the input of the model
        assert call[1][0].features is not None


@pytest.mark.elasticc
def skip_est_step_elasticc_without_features(step_factory_toretto):
    empty_features = []
    for msg in generate_messages_elasticc():
        msg.pop("features")
        empty_features.append(msg)
    step = step_factory_toretto(empty_features)
    step.start()
    scribe_calls = step.scribe_producer.mock_calls
    scribe_calls = step.scribe_producer.mock_calls
    predictor_calls = step.model.predict.mock_calls
    assert len(predictor_calls) == 0
    # Tests scribe produces correct commands
    assert len(scribe_calls) == 0
    for call in scribe_calls:
        message = loads(call.args[0]["payload"])
        assert_command_is_correct(message)
    # Test producer produces correct data
    calls = step.producer.mock_calls
    assert len(calls) == len(empty_features)
    for call in calls:
        obj = call.args[0]
        assert_elasticc_object_is_correct(obj)
