import json
from typing import Callable

import pytest
from apf.consumers import KafkaConsumer

from lc_classification.core.step import LateClassifier
from tests.test_commons import (
    assert_command_is_correct,
    assert_ztf_object_is_correct,
)


@pytest.mark.ztf
def test_step_ztf_result(
    kafka_service,
    produce_messages,
    env_variables_ztf,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_ztf")
    env_variables_ztf()

    from settings import config

    kconsumer = kafka_consumer("ztf")
    sconsumer = scribe_consumer()

    step = LateClassifier(config=config())
    step.start()

    for message in kconsumer.consume():
        assert_ztf_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        sconsumer.commit()


@pytest.mark.ztf
def test_step_ztf_no_features_result(
    kafka_service,
    produce_messages,
    env_variables_ztf,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_ztf", force_missing_features=True)
    env_variables_ztf()

    from settings import config

    kconsumer = kafka_consumer("ztf")
    sconsumer = scribe_consumer()

    step = LateClassifier(config=config())
    step.start()

    for message in kconsumer.consume():
        assert_ztf_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        sconsumer.commit()
