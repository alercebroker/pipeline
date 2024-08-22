import json
import os
from typing import Callable

import pytest
from unittest import mock
from apf.consumers import KafkaConsumer

from lc_classification.core.step import LateClassifier
from tests.test_commons import (
    assert_command_is_correct,
    assert_ztf_object_is_correct,
)


@pytest.mark.ztf
def test_step_squidward_result(
    kafka_service,
    produce_messages,
    env_variables_squidward,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_squidward")
    env_variables_squidward(
        "squidward",
        "alerce_classifiers.squidward.model.SquidwardFeaturesClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_SQUIDWARD_MODEL_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.squidward.mapper.SquidwardMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("squidward")
    sconsumer = scribe_consumer("w_object_squidward")

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
def test_step_squidward_no_features_result(
    kafka_service,
    produce_messages,
    env_variables_squidward,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_squidward", force_missing_features=True)
    env_variables_squidward(
        "squidward",
        "alerce_classifiers.squidward.model.SquidwardFeaturesClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_SQUIDWARD_MODEL_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.squidward.mapper.SquidwardMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("squidward")
    sconsumer = scribe_consumer("w_object_squidward")

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
def test_step_squidward_min_detections(
    kafka_service,
    produce_messages,
    env_variables_squidward,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):

    env_variables_squidward(
        "squidward",
        "alerce_classifiers.squidward.model.SquidwardFeaturesClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_SQUIDWARD_MODEL_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.squidward.mapper.SquidwardMapper",
            "MIN_DETECTIONS": "6",
        },
    )

    from settings import config

    produce_messages("features_squidward", n_forced=6)
    step = LateClassifier(config=config())
    step.execute = mock.MagicMock()
    step.start()
    step.execute.assert_not_called()

    produce_messages("features_squidward", n_forced=2)
    kconsumer = kafka_consumer("squidward")
    step = LateClassifier(config=config())
    step.start()

    for message in kconsumer.consume():
        assert_ztf_object_is_correct(message)
        kconsumer.commit()
