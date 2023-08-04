import os
from json import loads
from unittest.mock import MagicMock

import pytest
from fastavro import utils

from lc_classification.core.step import LateClassifier
from tests.mockdata.input_ztf import INPUT_SCHEMA as INPUT_ZTF
from tests.test_commons import assert_command_is_correct, assert_ztf_object_is_correct

messages_ztf = list(utils.generate_many(INPUT_ZTF, 2))

for message in messages_ztf:
    for det in message["detections"]:
        det["aid"] = message["aid"]
        det["extra_fields"] = {}
    message["detections"][0]["new"] = True
    message["detections"][0]["has_stamp"] = True


@pytest.mark.ztf
def test_step(step_factory_ztf):
    step: LateClassifier = step_factory_ztf(messages_ztf)
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
        obj = call.args[0]
        assert_ztf_object_is_correct(obj)


@pytest.mark.ztf
def test_step_empty_features(step_factory_ztf):
    empty_features = []
    for msg in messages_ztf:
        msg.pop("features")
        empty_features.append(msg)
    step = step_factory_ztf(empty_features)
    step.start()
    scribe_calls = step.scribe_producer.mock_calls
    assert scribe_calls == []
    calls = step.producer.mock_calls
    assert calls == []
