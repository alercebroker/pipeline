from apf.core import get_class
from apf.core.step import GenericStep
from lc_classification.core.parsers.kafka_parser import KafkaParser
import logging
import json
import numexpr
from alerce_classifiers.base.dto import OutputDTO
from lc_classification.core.parsers.input_dto import create_input_dto
from typing import List, Tuple
from pandas import DataFrame

ZTF_CLASSIFIER_CLASS = "lc_classifier.classifier.models.HierarchicalRandomForest"


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

    def __init__(self, config={}, level=logging.INFO, model=None, **step_args):
        super().__init__(config=config, level=level, **step_args)
        numexpr.utils.set_num_threads(1)

        self.isztf = config["MODEL_CONFIG"]["CLASS"] == ZTF_CLASSIFIER_CLASS
        self.logger.info("Loading Models")

        if model:
            self.model = model

        else:
            if self.isztf:
                self.model = get_class(config["MODEL_CONFIG"]["CLASS"])()
                self.model.download_model()
                self.model.load_model(self.model.MODEL_PICKLE_PATH)
            else:
                self.model = get_class(config["MODEL_CONFIG"]["CLASS"])(
                    **config["MODEL_CONFIG"]["PARAMS"]
                )

        self.scribe_producer = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])(
            config["SCRIBE_PRODUCER_CONFIG"]
        )
        self.scribe_parser: KafkaParser = get_class(config["SCRIBE_PARSER_CLASS"])()
        self.step_parser: KafkaParser = get_class(config["STEP_PARSER_CLASS"])()

        self.classifier_name = self.config["MODEL_CONFIG"]["NAME"]
        self.classifier_version = self.config["MODEL_VERSION"]

    def pre_produce(self, result: Tuple[OutputDTO, List[dict], DataFrame]):
        return self.step_parser.parse(
            result[0],
            messages=result[1],
            features=result[2],
            classifier_name=self.classifier_name,
            classifier_version=self.classifier_version,
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
        model_input = create_input_dto(messages)

        self.logger.info("Doing inference")
        if self.isztf:
            probabilities = self.model.predict_in_pipeline(
                model_input.features,
            )
            probabilities = OutputDTO(
                probabilities["probabilities"], probabilities["hierarchical"]
            )
        else:
            probabilities = self.model.predict(model_input)
        # after the former line, probabilities must be an OutputDTO

        self.logger.info("Processing results")
        return probabilities, messages, model_input.features

    def post_execute(self, result: Tuple[OutputDTO, List[dict]]):
        parsed_result = self.scribe_parser.parse(
            result[0], classifier_version=self.classifier_version
        )
        self.produce_scribe(parsed_result.value)
        return result
