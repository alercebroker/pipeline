from json import loads

import pytest
from fastavro import utils

from tests.mockdata.extra_felds import generate_extra_fields
from tests.mockdata.input_elasticc import INPUT_SCHEMA as INPUT_ELASTICC
from tests.test_commons import (
    assert_command_is_correct,
    assert_elasticc_object_is_correct,
)

messages_elasticc = list(utils.generate_many(INPUT_ELASTICC, 2))

for message in messages_elasticc:
    for det in message["detections"]:
        det["aid"] = message["aid"]
        det["extra_fields"] = generate_extra_fields()
    message["detections"][0]["new"] = True
    message["detections"][0]["has_stamp"] = True


@pytest.mark.elasticc
def test_step_toretto(test_elasticc_model, step_factory_toretto):
    test_elasticc_model(step_factory_toretto, messages_elasticc)


@pytest.mark.elasticc
def test_step_toretto_model_input_is_correct(step_factory_toretto):
    step = step_factory_toretto(messages_elasticc)
    step.start()
    predictor_calls = step.predictor.model.predict.mock_calls
    assert len(predictor_calls) > 0
    for call in predictor_calls:
        # check that there are features in the input of the model
        assert call[1][0].features.any().any()


@pytest.mark.elasticc
def test_step_elasticc_without_features(step_factory_toretto):
    empty_features = []
    for msg in messages_elasticc:
        msg.pop("features")
        empty_features.append(msg)
    step = step_factory_toretto(empty_features)
    step.start()
    scribe_calls = step.scribe_producer.mock_calls
    scribe_calls = step.scribe_producer.mock_calls
    predictor_calls = step.predictor.model.predict.mock_calls
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
