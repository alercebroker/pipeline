import json
from lc_classification.core.step import LateClassifier
from apf.consumers import KafkaConsumer
import pytest
import os
from tests.test_commons import (
    assert_elasticc_object_is_correct,
    assert_command_is_correct,
)


@pytest.mark.skipif(os.getenv("STREAM") != "elasticc", reason="elasticc only")
def test_step_elasticc_result(
    kafka_service,
    env_variables_elasticc,
    kafka_consumer: KafkaConsumer,
    scribe_consumer: KafkaConsumer,
):
    from settings import STEP_CONFIG

    model_path = "https://assets.alerce.online/pipeline/elasticc/random_forest/2.0.1/"
    STEP_CONFIG["PREDICTOR_CONFIG"][
        "CLASS"
    ] = "lc_classification.predictors.toretto.toretto_predictor.TorettoPredictor"
    STEP_CONFIG["PREDICTOR_CONFIG"]["PARAMS"] = {"model_path": model_path}
    STEP_CONFIG["PREDICTOR_CONFIG"][
        "PARSER_CLASS"
    ] = "lc_classification.predictors.toretto.toretto_parser.TorettoParser"
    STEP_CONFIG[
        "SCRIBE_PARSER_CLASS"
    ] = "lc_classification.core.parsers.scribe_parser.ScribeParser"
    STEP_CONFIG[
        "STEP_PARSER_CLASS"
    ] = "lc_classification.core.parsers.elasticc_parser.ElasticcParser"
    step = LateClassifier(config=STEP_CONFIG)
    step.start()

    for message in kafka_consumer.consume():
        assert_elasticc_object_is_correct(message)
        kafka_consumer.commit()

    for message in scribe_consumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        scribe_consumer.commit()
