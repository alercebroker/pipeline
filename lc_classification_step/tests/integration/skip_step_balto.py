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
        "balto",
        "alerce_classifiers.balto.model.BaltoClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_BALTO_MODEL_PATH"),
            "QUANTILES_PATH": os.getenv("TEST_BALTO_QUANTILES_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.balto.mapper.BaltoMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("balto")
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
def test_step_schemaless(
    kafka_service,
    env_variables_elasticc,
    kafka_consumer: Callable[[str, str, dict], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    env_variables_elasticc(
        "balto_schemaless",
        "alerce_classifiers.balto.model.BaltoClassifier",
        {
            "MODEL_PATH": os.getenv("TEST_BALTO_MODEL_PATH"),
            "QUANTILES_PATH": os.getenv("TEST_BALTO_QUANTILES_PATH"),
            "MAPPER_CLASS": "alerce_classifiers.balto.mapper.BaltoMapper",
            "PRODUCER_CLASS": "apf.producers.kafka.KafkaSchemalessProducer",
        },
    )

    from settings import config

    kconsumer = kafka_consumer(
        "balto_schemaless",
        "apf.consumers.kafka.KafkaSchemalessConsumer",
        {
            "SCHEMA_PATH": "../schemas/lc_classification_step/output_elasticc.avsc"
        },
    )
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
