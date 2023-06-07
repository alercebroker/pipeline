from json import loads
from tests.mockdata.input_elasticc import INPUT_SCHEMA as INPUT_ELASTICC
from fastavro import utils
import pytest
import os

from tests.test_commons import (
    assert_elasticc_object_is_correct,
    assert_command_is_correct,
)

messages_elasticc = list(utils.generate_many(INPUT_ELASTICC, 10))

for message in messages_elasticc:
    for det in message["detections"]:
        det["aid"] = message["aid"]
    message["detections"][0]["new"] = True
    message["detections"][0]["has_stamp"] = True


@pytest.mark.skipif(os.getenv("STREAM") != "elasticc", reason="elasticc only")
def test_step_toretto(step_factory_toretto):
    step = step_factory_toretto(messages_elasticc)
    step.start()
    scribe_calls = step.scribe_producer.mock_calls
    predictor_calls = step.predictor.model.predict.mock_calls
    assert len(predictor_calls) > 0
    for call in predictor_calls:
        # check that there are features in the input of the model
        assert call[1][0].features.any().any()
    # Tests scribe produces correct commands
    for call in scribe_calls:
        message = loads(call.args[0]["payload"])
        assert_command_is_correct(message)
    # Test producer produces correct data
    calls = step.producer.mock_calls
    assert len(calls) == len(messages_elasticc)
    for call in calls:
        obj = call.args[0]
        assert_elasticc_object_is_correct(obj)


@pytest.mark.skipif(os.getenv("STREAM") != "elasticc", reason="elasticc only")
def test_step_elasticc_without_features(step_factory_toretto):
    empty_features = []
    for msg in messages_elasticc:
        msg["features"] = None
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
