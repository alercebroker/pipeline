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
    env_variables_elasticc("balto")

    from settings import STEP_CONFIG

    kconsumer = kafka_consumer("balto")
    sconsumer = scribe_consumer()

    model_path = os.getenv("TEST_BALTO_MODEL_PATH")
    quantiles_path = os.getenv("TEST_BALTO_QUANTILES_PATH")

    STEP_CONFIG["PREDICTOR_CONFIG"][
        "CLASS"
    ] = "lc_classification.predictors.balto.balto_predictor.BaltoPredictor"
    STEP_CONFIG["PREDICTOR_CONFIG"]["PARAMS"] = {
        "model_path": model_path,
        "quantiles_path": quantiles_path,
    }
    STEP_CONFIG["PREDICTOR_CONFIG"][
        "PARSER_CLASS"
    ] = "lc_classification.predictors.balto.balto_parser.BaltoParser"
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
