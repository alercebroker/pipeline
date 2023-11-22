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
def test_step_elasticc_result(
    kafka_service,
    produce_messages,
    env_variables_elasticc,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_elasticc")
    env_variables_elasticc(
        "messi",
        "alerce_classifiers.messi.model.MessiClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MESSI_MODEL_PATH"),
            "BALTO_MODEL_PATH": os.getenv("TEST_BALTO_MODEL_PATH"),
            "HEADER_QUANTILES_PATH": os.getenv(
                "TEST_MESSI_HEADER_QUANTILES_PATH"
            ),
            "FEATURE_QUANTILES_PATH": os.getenv(
                "TEST_MESSI_FEATURE_QUANTILES_PATH"
            ),
            "MAPPER_CLASS": "alerce_classifiers.messi.mapper.MessiMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("messi")
    sconsumer = scribe_consumer()

    step = LateClassifier(config=config())
    step.start()

    for message in kconsumer.consume():
        assert_elasticc_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        sconsumer.commit()


@pytest.mark.elasticc
def test_step_elasticc_result_empty_features(
    kafka_service,
    produce_messages,
    env_variables_elasticc,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_elasticc", force_empty_features=True)
    env_variables_elasticc(
        "messi",
        "alerce_classifiers.messi.model.MessiClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MESSI_MODEL_PATH"),
            "BALTO_MODEL_PATH": os.getenv("TEST_BALTO_MODEL_PATH"),
            "HEADER_QUANTILES_PATH": os.getenv(
                "TEST_MESSI_HEADER_QUANTILES_PATH"
            ),
            "FEATURE_QUANTILES_PATH": os.getenv(
                "TEST_MESSI_FEATURE_QUANTILES_PATH"
            ),
            "MAPPER_CLASS": "alerce_classifiers.messi.mapper.MessiMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("messi")
    sconsumer = scribe_consumer()

    step = LateClassifier(config=config())
    step.start()

    for message in kconsumer.consume():
        assert_elasticc_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        sconsumer.commit()


@pytest.mark.elasticc
def test_step_elasticc_result_missing_features(
    kafka_service,
    produce_messages,
    env_variables_elasticc,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_elasticc", force_missing_features=True)
    env_variables_elasticc(
        "messi",
        "alerce_classifiers.messi.model.MessiClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MESSI_MODEL_PATH"),
            "BALTO_MODEL_PATH": os.getenv("TEST_BALTO_MODEL_PATH"),
            "HEADER_QUANTILES_PATH": os.getenv(
                "TEST_MESSI_HEADER_QUANTILES_PATH"
            ),
            "FEATURE_QUANTILES_PATH": os.getenv(
                "TEST_MESSI_FEATURE_QUANTILES_PATH"
            ),
            "MAPPER_CLASS": "alerce_classifiers.messi.mapper.MessiMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("messi")
    sconsumer = scribe_consumer()

    step = LateClassifier(config=config())
    step.start()

    for message in kconsumer.consume():
        assert_elasticc_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        sconsumer.commit()
