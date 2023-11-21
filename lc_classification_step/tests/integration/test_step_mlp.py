import json
import os
from typing import Callable

import pytest
from apf.consumers import KafkaConsumer

from lc_classification.core.step import LateClassifier
from tests.test_commons import (
    assert_command_is_correct,
    assert_elasticc_object_is_correct,
)


@pytest.mark.elasticc
def test_step_elasticc_result_mlp(
    kafka_service,
    env_variables_elasticc,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    env_variables_elasticc(
        "mlp",
        "alerce_classifiers.tinkywinky.model.TinkyWinkyClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MLP_MODEL_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.mlp_elasticc.mapper.MLPMapper",
        },
    )

    from settings import settings_creator

    kconsumer = kafka_consumer("mlp")
    sconsumer = scribe_consumer()

    step = LateClassifier(config=settings_creator())
    step.start()

    for message in kconsumer.consume():
        assert_elasticc_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        sconsumer.commit()


@pytest.mark.elasticc
def test_step_elasticc_result_mlp_without_features(
    kafka_service,
    env_variables_elasticc,
    produce_messages,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_elasticc", force_empty_features=True)
    env_variables_elasticc(
        "mlp",
        "alerce_classifiers.tinkywinky.model.TinkyWinkyClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MLP_MODEL_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.mlp_elasticc.mapper.MLPMapper",
        },
    )

    from settings import settings_creator

    kconsumer = kafka_consumer("mlp")
    sconsumer = scribe_consumer()

    step = LateClassifier(config=settings_creator())
    step.start()

    for message in kconsumer.consume():
        assert_elasticc_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        sconsumer.commit()
