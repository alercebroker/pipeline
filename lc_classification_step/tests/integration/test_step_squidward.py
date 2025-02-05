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
def test_step_squidward_min_detections_greater(
    env_variables_squidward,
):
    env_variables_squidward(
        "squidward",
        "alerce_classifiers.squidward.model.SquidwardFeaturesClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_SQUIDWARD_MODEL_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.squidward.mapper.SquidwardMapper",
            "MIN_DETECTIONS": "2",
        },
    )

    from settings import config
    from .conftest import INPUT_SCHEMA_PATH, add_fields_to_message
    from fastavro.utils import generate_many
    from fastavro.schema import load_schema
    import random

    random.seed(42)
    schema = load_schema(str(INPUT_SCHEMA_PATH))
    messages = generate_many(schema, 2)  # 1 object has features
    messages = list(messages)

    for message in messages:
        message = add_fields_to_message(
            message,
            "features_ztf",
            ["g", "r"],
            False,
            False,
            5,
        )
        for det in message["detections"]:
            if not det["forced"]:
                det["extra_fields"]["rb"] = 0.8

    step = LateClassifier(config=config())

    output, result_messages, features = step.execute(messages)
    probabilities = output.probabilities

    assert len(probabilities) == 1


@pytest.mark.ztf
def test_step_squidward_min_detections_lower(
    env_variables_squidward,
):
    env_variables_squidward(
        "squidward",
        "alerce_classifiers.squidward.model.SquidwardFeaturesClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_SQUIDWARD_MODEL_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.squidward.mapper.SquidwardMapper",
            "MIN_DETECTIONS": "8",
        },
    )

    from settings import config
    from .conftest import INPUT_SCHEMA_PATH, add_fields_to_message
    from fastavro.utils import generate_many
    from fastavro.schema import load_schema
    import random

    random.seed(42)
    schema = load_schema(str(INPUT_SCHEMA_PATH))
    messages = generate_many(schema, 2)
    messages = list(messages)

    for message in messages:
        message = add_fields_to_message(
            message,
            "features_ztf",
            ["g", "r"],
            False,
            False,
            5,
        )
        for det in message["detections"]:
            if not det["forced"]:
                det["extra_fields"]["rb"] = 0.8

    step = LateClassifier(config=config())

    output, result_messages, features = step.execute(messages)
    probabilities = output.probabilities

    assert len(probabilities) == 0
