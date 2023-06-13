import json
from lc_classification.core.step import LateClassifier
from apf.consumers import KafkaConsumer
import pytest
import os
from typing import Callable
from tests.test_commons import assert_command_is_correct, assert_ztf_object_is_correct


@pytest.mark.skipif(os.getenv("STREAM") != "ztf", reason="ztf only")
def test_step_ztf_result(
    kafka_service,
    env_variables_ztf,
    kafka_consumer: Callable[[], KafkaConsumer],
    scribe_consumer: Callable[[], KafkaConsumer],
):
    from settings import STEP_CONFIG

    kconsumer = kafka_consumer()
    sconsumer = scribe_consumer()

    STEP_CONFIG["PREDICTOR_CONFIG"][
        "CLASS"
    ] = "lc_classification.predictors.ztf_random_forest.ztf_random_forest_predictor.ZtfRandomForestPredictor"
    STEP_CONFIG["PREDICTOR_CONFIG"][
        "PARSER_CLASS"
    ] = "lc_classification.predictors.ztf_random_forest.ztf_random_forest_parser.ZtfRandomForestParser"
    STEP_CONFIG[
        "SCRIBE_PARSER_CLASS"
    ] = "lc_classification.core.parsers.scribe_parser.ScribeParser"
    STEP_CONFIG[
        "STEP_PARSER_CLASS"
    ] = "lc_classification.core.parsers.alerce_parser.AlerceParser"
    STEP_CONFIG["PREDICTOR_CONFIG"]["PARAMS"] = {}
    step = LateClassifier(config=STEP_CONFIG)
    step.start()

    for message in kconsumer.consume():
        assert_ztf_object_is_correct(message)
        kconsumer.commit()

    for message in sconsumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        sconsumer.commit()
