import json
import os
from typing import Callable

import pytest
from apf.consumers import KafkaConsumer

from lc_classification.core.step import LateClassifier
from tests.test_commons import (
    assert_score_command_is_correct,
    assert_ztf_object_is_correct,
)


@pytest.mark.ztf
def test_step_anomaly_result(
    kafka_service,
    produce_messages,
    env_variables_anomaly,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_anomaly")
    env_variables_anomaly(
        "anomaly",
        "alerce_classifiers.anomaly.model.AnomalyDetector",
        {
            "MODEL_PATH": os.getenv("TEST_ANOMALY_MODEL_PATH"), 
            "FEATURE_QUANTILES_PATH": os.getenv(
                "TEST_ANOMALY_QUANTILES_PATH"
            ),
            "MAPPER_CLASS": "alerce_classifiers.anomaly.mapper.AnomalyMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("anomaly")
    sconsumer = scribe_consumer("w_object_anomaly")

    step = LateClassifier(config=config())
    step.start()

    for message in kconsumer.consume():
        assert_ztf_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_score_command_is_correct(command)
        sconsumer.commit()


@pytest.mark.ztf
def test_step_anomaly_no_features_result(
    kafka_service,
    produce_messages,
    env_variables_anomaly,
    kafka_consumer: Callable[[str], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    produce_messages("features_anomaly", force_missing_features=True)
    env_variables_anomaly(
        "anomaly",
        "alerce_classifiers.anomaly.model.AnomalyDetector",
        {
            "MODEL_PATH": os.getenv("TEST_ANOMALY_MODEL_PATH"),
            "FEATURE_QUANTILES_PATH": os.getenv(
                "TEST_ANOMALY_QUANTILES_PATH"
            ),
            "MAPPER_CLASS": "alerce_classifiers.anomaly.mapper.AnomalyMapper",
        },
    )

    from settings import config

    kconsumer = kafka_consumer("anomaly")
    sconsumer = scribe_consumer("w_object_anomaly")

    step = LateClassifier(config=config())
    step.start()

    for message in kconsumer.consume():
        assert_ztf_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_score_command_is_correct(command)
        sconsumer.commit()
