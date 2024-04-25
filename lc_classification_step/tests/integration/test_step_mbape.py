import json
import os
from typing import Callable

import pytest
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
        "mbape",
        "alerce_classifiers.mbappe.model.MbappeClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MBAPPE_MODEL_PATH"), 
            "FEATURE_QUANTILES_PATH": os.getenv(
                "TEST_MBAPPE_FEATURE_QUANTILES_PATH"
            ),
            "HEADER_QUANTILES_PATH": os.getenv(
                "TEST_MBAPPE_HEADER_QUANTILES_PATH"
            ),
            "MAPPER_CLASS": "alerce_classifiers.mbappe.mapper.MbappeMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("mbappe")
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
def test_step_mbappe_no_features_result(
    kafka_service,
    produce_messages,
    env_variables_mbappe,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_mbappe", force_missing_features=True)
    env_variables_mbappe(
        "mbape",
        "alerce_classifiers.mbappe.model.MbappeClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_MBAPPE_MODEL_PATH"), 
            "FEATURE_QUANTILES_PATH": os.getenv(
                "TEST_MBAPPE_FEATURE_QUANTILES_PATH"
            ),
            "HEADER_QUANTILES_PATH": os.getenv(
                "TEST_MBAPPE_HEADER_QUANTILES_PATH"
            ),
            "MAPPER_CLASS": "alerce_classifiers.mbappe.mapper.MbappeMapper",
        },
    )

    from settings import config 

    kconsumer = kafka_consumer("mbappe")
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
