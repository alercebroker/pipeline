from typing import List, Union
from apf.core import get_class
from apf.core.step import GenericStep
from lc_classifier.classifier.models import HierarchicalRandomForest
from lc_classification.core.kafka_parser import KafkaParser
from lc_classification.core.step_parser import StepParser
from lc_classification.core.scribe_parser import ScribeParser

import logging
import json

import numexpr
from lc_classification.predictors.predictor.predictor_parser import PredictorParser

from lc_classification.predictors.ztf_random_forest.ztf_random_forest_parser import (
    ZtfRandomForestPredictorParser,
)


class LateClassifier(GenericStep):
    """Light Curve Classification Step, for a description of the algorithm used to process
    check the `execute()` method.

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    base_name = "lc_classifier"

    def __init__(
        self,
        config={},
        level=logging.INFO,
        model=None,
        scribe_parser: KafkaParser = ScribeParser(),
        step_parser: KafkaParser = StepParser(),
        predictor_parser: Union[PredictorParser, None] = None,
        **step_args
    ):
        super().__init__(config=config, level=level, **step_args)
        numexpr.utils.set_num_threads(1)
        self.logger.info("Loading Models")
        self.model = model or HierarchicalRandomForest({})
        self.model.download_model()
        self.model.load_model(self.model.MODEL_PICKLE_PATH)
        cls = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(config["SCRIBE_PRODUCER_CONFIG"])
        self.predictor_parser = predictor_parser or ZtfRandomForestPredictorParser(
            self.model.feature_list
        )
        self.scribe_parser = scribe_parser
        self.step_parser = step_parser

    def pre_produce(self, result: tuple):
        return self.step_parser.parse(
            result[0], messages=result[1], features=result[2]
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
        tree_probabilities = self.model.predict_in_pipeline(predictor_input.value)
        self.logger.info("Processing results")
        predictor_output = self.predictor_parser.parse_output(tree_probabilities)
        return {
            "public_info": (predictor_output, messages, predictor_input.value),
            "db_results": self.scribe_parser.parse(predictor_output),
        }

    def post_execute(self, result: dict):
        db_results = result.pop("db_results")
        self.produce_scribe(db_results.value)
        return result["public_info"]
