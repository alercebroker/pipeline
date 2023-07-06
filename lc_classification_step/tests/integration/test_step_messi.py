import json
import os
from typing import Callable

from apf.consumers import KafkaConsumer
from lc_classification.core.step import LateClassifier
from schemas import ELASTICC_SCHEMA
from tests.test_commons import (
    assert_command_is_correct,
    assert_elasticc_object_is_correct,
)


def test_step_elasticc_result(
    kafka_service,
    env_variables_elasticc,
    kafka_consumer: Callable[[], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    env_variables_elasticc("messi")

    from settings import STEP_CONFIG

    kconsumer = kafka_consumer("messi")
    sconsumer = scribe_consumer()

    model_path = os.getenv("TEST_MESSI_MODEL_PATH")
    header_quantiles_path = os.getenv("TEST_MESSI_HEADER_QUANTILES_PATH")
    feature_quantiles_path = os.getenv("TEST_MESSI_FEATURE_QUANTILES_PATH")

    STEP_CONFIG["PREDICTOR_CONFIG"][
        "CLASS"
    ] = "lc_classification.predictors.messi.messi_predictor.MessiPredictor"
    STEP_CONFIG["PREDICTOR_CONFIG"]["PARAMS"] = {
        "model_path": model_path,
        "header_quantiles_path": header_quantiles_path,
        "feature_quantiles_path": feature_quantiles_path,
    }
    STEP_CONFIG["PREDICTOR_CONFIG"][
        "PARSER_CLASS"
    ] = "lc_classification.predictors.messi.messi_parser.MessiParser"
    STEP_CONFIG[
        "SCRIBE_PARSER_CLASS"
    ] = "lc_classification.core.parsers.scribe_parser.ScribeParser"
    STEP_CONFIG[
        "STEP_PARSER_CLASS"
    ] = "lc_classification.core.parsers.elasticc_parser.ElasticcParser"
    step = LateClassifier(config=STEP_CONFIG)
    step.start()

    for message in kconsumer.consume():
        assert_elasticc_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        sconsumer.commit()
