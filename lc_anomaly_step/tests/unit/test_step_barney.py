from json import loads

import pytest
from tests.test_commons import (
    assert_command_is_correct,
    assert_elasticc_object_is_correct,
)
from tests.unit.test_utils import generate_messages_elasticc


def assert_classified_as_not_classified(step, obj):
    class_mapper = step.step_parser.ClassMapper
    not_classified_id = class_mapper.get_class_value("NotClassified")
    not_classified_probability = next(
        classification["probability"]
        for classification in obj["classifications"]
        if classification["classId"] == not_classified_id
    )

    assert not_classified_probability == 1


@pytest.mark.elasticc
def test_step_barney(test_elasticc_model, step_factory_barney):
    test_elasticc_model(step_factory_barney, generate_messages_elasticc())


@pytest.mark.elasticc
def test_step_braney_model_input_is_correct(step_factory_barney):
    step = step_factory_barney(generate_messages_elasticc())
    step.start()
    predictor_calls = step.model.predict.mock_calls
    assert len(predictor_calls) > 0
    for call in predictor_calls:
        # check that there are features in the input of the model
        assert call[1][0] is not None


@pytest.mark.elasticc
def stkip_test_step_elasticc_without_features(step_factory_barney):
    empty_features = []
    for msg in generate_messages_elasticc():
        msg.pop("features")
        empty_features.append(msg)
    step = step_factory_barney(empty_features)
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
        assert_classified_as_not_classified(step, obj)
        assert_elasticc_object_is_correct(obj)
