from typing import List
from apf.core import get_class
from apf.core.step import GenericStep
from lc_classification.core.parsers.kafka_parser import KafkaParser
import logging
import json

import numexpr
from lc_classification.predictors.predictor.predictor import Predictor
from lc_classification.predictors.predictor.predictor_parser import PredictorParser


class LateClassifier(GenericStep):
    """Light Curve Classification Step, for a description of the algorithm used
    to process check the `execute()` method.

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    base_name = "lc_classifier"

    def __init__(self, config={}, level=logging.INFO, **step_args):
        super().__init__(config=config, level=level, **step_args)
        numexpr.utils.set_num_threads(1)
        self.logger.info("Loading Models")
        scribe_producer_class = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        if scribe_producer_class == "lc_classification.predictors.ztf_random_forest.ztf_random_forest_predictor.ZtfRandomForestPredictor":
            self.predictor: Predictor = get_class(config["PREDICTOR_CONFIG"]["CLASS"])(
                **config["PREDICTOR_CONFIG"]["PARAMS"]
            )
            self.scribe_producer = scribe_producer_class(config["SCRIBE_PRODUCER_CONFIG"])
            self.predictor_parser: PredictorParser = get_class(
                config["PREDICTOR_CONFIG"]["PARSER_CLASS"]
            )()
            self.scribe_parser: KafkaParser = get_class(config["SCRIBE_PARSER_CLASS"])()
            self.step_parser: KafkaParser = get_class(config["STEP_PARSER_CLASS"])()
        else:
            self.predictor: Predictor = get_class(config["PREDICTOR_CONFIG"]["CLASS"])(
                **config["PREDICTOR_CONFIG"]["PARAMS"]
            )
            self.scribe_producer = scribe_producer_class(config["SCRIBE_PRODUCER_CONFIG"])
            self.predictor_parser: PredictorParser = get_class(
                config["PREDICTOR_CONFIG"]["PARSER_CLASS"]
            )()
            self.scribe_parser: KafkaParser = get_class(config["SCRIBE_PARSER_CLASS"])()
            self.step_parser: KafkaParser = get_class(config["STEP_PARSER_CLASS"])()

    def pre_produce(self, result: tuple):
        return self.step_parser.parse(
            result[0],
            messages=result[1],
            features=result[2],
            classifier_name=self.predictor.__class__.__name__,
            classifier_version=self.config["MODEL_VERSION"],
        ).value

    def produce_scribe(self, commands: List[dict]):
        for command in commands:
            self.scribe_producer.produce({"payload": json.dumps(command)})

    def execute(self, messages):
        """Run the classification.

        Parameters
        ----------
        messages : dict-like
            Current object data, it must have the features and object id.

        """
        self.logger.info("Processing %i messages.", len(messages))
        self.logger.info("Getting batch alert data")
        predictor_input = self.predictor_parser.parse_input(messages)
        self.logger.info("Doing inference")
        probabilities = self.predictor.predict(predictor_input)
        self.logger.info("Processing results")
        predictor_output = self.predictor_parser.parse_output(probabilities)
        return {
            "public_info": (predictor_output, messages, predictor_input.value),
            "db_results": self.scribe_parser.parse(
                predictor_output, classifier_version=self.config["MODEL_VERSION"]
            ),
        }

    def post_execute(self, result: dict):
        db_results = result.pop("db_results")
        self.produce_scribe(db_results.value)
        return result["public_info"]
