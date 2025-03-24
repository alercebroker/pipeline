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
def test_step_mbappe_result(
    kafka_service,
    produce_messages,
    env_variables_mbappe,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_mbappe")
    env_variables_mbappe(
        "mbappe",
        "alerce_classifiers.mbappe.model.MbappeClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MBAPPE_MODEL_PATH"),
            "QUANTILES_PATH": os.getenv("TEST_MBAPPE_QUANTILES_PATH"),
            "CONFIG_PATH": os.getenv("TEST_MBAPPE_CONFIG_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.mbappe.mapper.MbappeMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("mbappe")
    sconsumer = scribe_consumer("w_object_mbappe")

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
def test_step_mbappe_no_features_result(
    kafka_service,
    produce_messages,
    env_variables_mbappe,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_mbappe", force_missing_features=True)
    env_variables_mbappe(
        "mbappe",
        "alerce_classifiers.mbappe.model.MbappeClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MBAPPE_MODEL_PATH"),
            "QUANTILES_PATH": os.getenv("TEST_MBAPPE_QUANTILES_PATH"),
            "CONFIG_PATH": os.getenv("TEST_MBAPPE_CONFIG_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.mbappe.mapper.MbappeMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("mbappe")
    sconsumer = scribe_consumer("w_object_mbappe")

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
def test_step_mbappe_nonvalid_detections(
    env_variables_mbappe,
):
    env_variables_mbappe(
        "mbappe",
        "alerce_classifiers.mbappe.model.MbappeClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MBAPPE_MODEL_PATH"),
            "QUANTILES_PATH": os.getenv("TEST_MBAPPE_QUANTILES_PATH"),
            "CONFIG_PATH": os.getenv("TEST_MBAPPE_CONFIG_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.mbappe.mapper.MbappeMapper",
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
            "features_mbappe",
            ["g", "r"],
            False,
            False,
            2,
        )
        message["detections"][0]["e_mag"] = 100

    step = LateClassifier(config=config())

    output, result_messages, features = step.execute(messages)
    probabilities = output.probabilities

    for message in result_messages:
        assert len(message["detections"]) == 9

    assert len(probabilities) == 1

    del step


@pytest.mark.ztf
def test_step_mbappe_min_detections_lower(
    env_variables_mbappe,
):
    env_variables_mbappe(
        "mbappe",
        "alerce_classifiers.mbappe.model.MbappeClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MBAPPE_MODEL_PATH"),
            "QUANTILES_PATH": os.getenv("TEST_MBAPPE_QUANTILES_PATH"),
            "CONFIG_PATH": os.getenv("TEST_MBAPPE_CONFIG_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.mbappe.mapper.MbappeMapper",
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
            "features_mbappe",
            ["g", "r"],
            False,
            False,
            5,
        )

    step = LateClassifier(config=config())

    output, result_messages, features = step.execute(messages)
    probabilities = output.probabilities

    assert len(probabilities) == 0

    del step


@pytest.mark.ztf
def test_step_mbappe_min_detections_greater(
    kafka_service,
    produce_messages,
    env_variables_mbappe,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    env_variables_mbappe(
        "mbappe",
        "alerce_classifiers.mbappe.model.MbappeClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MBAPPE_MODEL_PATH"),
            "QUANTILES_PATH": os.getenv("TEST_MBAPPE_QUANTILES_PATH"),
            "CONFIG_PATH": os.getenv("TEST_MBAPPE_CONFIG_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.mbappe.mapper.MbappeMapper",
            "MIN_DETECTIONS": "6",
        },
    )

    from settings import config

    produce_messages("features_mbappe", n_forced=2)
    kconsumer = kafka_consumer("mbappe")
    step = LateClassifier(config=config())
    step.start()

    for message in kconsumer.consume():
        assert_ztf_object_is_correct(message)
        kconsumer.commit()
