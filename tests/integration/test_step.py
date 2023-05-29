import json
from lc_classification.core.step import LateClassifier
from apf.consumers import KafkaConsumer
import pytest
import os


def assert_object_is_correct(obj):
    assert "aid" in obj
    assert "candid" in obj
    assert "features" in obj
    assert "lc_classification" in obj


def assert_elasticc_object_is_correct(obj):
    assert "classifications" in obj
    assert isinstance(obj["classifications"], list)
    assert len(obj["classifications"]) > 0


def assert_command_is_correct(command):
    assert command["collection"] == "object"
    assert command["type"] == "update_probabilities"
    assert command["criteria"]["_id"] is not None
    assert "aid" not in command["data"]
    assert not command["options"]["set_on_insert"]


@pytest.mark.skipif(os.getenv("MODEL") != "ztf", reason="ztf only")
def test_step_ztf_result(
    kafka_service,
    env_variables_ztf,
    kafka_consumer: KafkaConsumer,
):
    from settings import STEP_CONFIG

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

    for message in kafka_consumer.consume():
        assert_object_is_correct(message)
        kafka_consumer.commit()


@pytest.mark.skipif(os.getenv("MODEL") != "ztf", reason="ztf only")
def test_scribe_ztf_result(
    kafka_service, env_variables_ztf, scribe_consumer: KafkaConsumer
):
    from settings import STEP_CONFIG

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

    for message in scribe_consumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        scribe_consumer.commit()


@pytest.mark.skipif(os.getenv("MODEL") != "elasticc", reason="elasticc only")
def test_step_elasticc_result(
    kafka_service,
    env_variables_elasticc,
    kafka_consumer: KafkaConsumer,
):
    from settings import STEP_CONFIG

    model_path = "https://assets.alerce.online/pipeline/elasticc/random_forest/2.0.1/"
    STEP_CONFIG["PREDICTOR_CONFIG"][
        "CLASS"
    ] = "lc_classification.predictors.elasticc_random_forest.elasticc_random_forest_predictor.ElasticcRandomForestPredictor"
    STEP_CONFIG["PREDICTOR_CONFIG"]["PARAMS"] = {"model_path": model_path}
    STEP_CONFIG["PREDICTOR_CONFIG"][
        "PARSER_CLASS"
    ] = "lc_classification.predictors.elasticc_random_forest.elasticc_random_forest_parser.ElasticcRandomForestParser"
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


@pytest.mark.skipif(os.getenv("MODEL") != "elasticc", reason="elasticc only")
def test_scribe_elasticc_result(
    kafka_service, env_variables_elasticc, scribe_consumer: KafkaConsumer
):
    from settings import STEP_CONFIG

    model_path = "https://assets.alerce.online/pipeline/elasticc/random_forest/2.0.1/"
    STEP_CONFIG["PREDICTOR_CONFIG"][
        "CLASS"
    ] = "lc_classification.predictors.elasticc_random_forest.elasticc_random_forest_predictor.ElasticcRandomForestPredictor"
    STEP_CONFIG["PREDICTOR_CONFIG"]["PARAMS"] = {"model_path": model_path}
    STEP_CONFIG["PREDICTOR_CONFIG"][
        "PARSER_CLASS"
    ] = "lc_classification.predictors.elasticc_random_forest.elasticc_random_forest_parser.ElasticcRandomForestParser"
    STEP_CONFIG[
        "SCRIBE_PARSER_CLASS"
    ] = "lc_classification.core.parsers.scribe_parser.ScribeParser"
    STEP_CONFIG[
        "STEP_PARSER_CLASS"
    ] = "lc_classification.core.parsers.elasticc_parser.ElasticcParser"
    step = LateClassifier(config=STEP_CONFIG)
    step.start()

    for message in scribe_consumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        scribe_consumer.commit()
