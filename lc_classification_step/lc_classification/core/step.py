from apf.core import get_class
from apf.consumers import KafkaConsumer
from apf.core.step import GenericStep
from lc_classification.core.parsers.kafka_parser import KafkaParser
import logging
import json
import numexpr
from alerce_classifiers.base.dto import OutputDTO
from lc_classification.core.parsers.input_dto import create_input_dto
from typing import List, Tuple
from pandas import DataFrame

ZTF_CLASSIFIER_CLASS = (
    "lc_classifier.classifier.models.HierarchicalRandomForest"
)


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
        self.classifier_name = self.config["MODEL_CONFIG"]["NAME"]
        numexpr.utils.set_num_threads(1)

        self.isztf = config["MODEL_CONFIG"]["CLASS"] == ZTF_CLASSIFIER_CLASS
        self.logger.info("Loading Models")

        if model:
            self.model = model
            self.classifier_version = self.config["MODEL_VERSION"]

        else:
            if self.isztf:
                self.model = get_class(config["MODEL_CONFIG"]["CLASS"])()
                self.model.download_model()
                self.model.load_model(self.model.MODEL_PICKLE_PATH)
                self.classifier_version = self.config["MODEL_VERSION"]
            else:
                mapper = get_class(
                    config["MODEL_CONFIG"]["PARAMS"]["mapper"]
                )()
                config["MODEL_CONFIG"]["PARAMS"]["mapper"] = mapper
                self.model = get_class(config["MODEL_CONFIG"]["CLASS"])(
                    **config["MODEL_CONFIG"]["PARAMS"]
                )
                self.classifier_version = self.model.model_version
        self.scribe_producer = get_class(
            config["SCRIBE_PRODUCER_CONFIG"]["CLASS"]
        )(config["SCRIBE_PRODUCER_CONFIG"])
        self.scribe_parser: KafkaParser = get_class(
            config["SCRIBE_PARSER_CLASS"]
        )(classifier_name=self.classifier_name)
        self.step_parser: KafkaParser = get_class(
            config["STEP_PARSER_CLASS"]
        )()
        self.min_detections = config.get("MIN_DETECTIONS", None)
        if self.min_detections is not None:
            self.min_detections = int(self.min_detections)

    def pre_produce(self, result: Tuple[OutputDTO, List[dict], DataFrame]):
        return self.step_parser.parse(
            result[0],
            messages=result[1],
            features=result[2],
            classifier_name=self.classifier_name,
            classifier_version=self.classifier_version,
        ).value

    def produce_scribe(self, commands: List[dict]):
        ids_list = []
        for command in commands:
            ids_list.append(command["criteria"]["_id"])
            self.scribe_producer.produce({"payload": json.dumps(command)})
        self.logger.debug(f"The list of objets from scribe are: {ids_list}")

    def log_data(self, model_input):
        forced = []
        prv_candidates = []
        dia_object = []
        for det in model_input._detections._value.iterrows():
            if det[1]["forced"]:
                forced.append(det[0])
                if "diaObject" in det[1].index:
                    dia_object.append(det[0])
                if det[1]["parent_candid"] is not None:
                    prv_candidates.append(det[0])
                if "diaObject" in det[1].index:
                    dia_object.append(det[0])
        self.logger.info(
            "The number of detections is: %i",
            len(model_input.detections),
        )
        self.logger.debug(f"The forced photometry detections are: {forced}")
        self.logger.debug(
            f"The prv candidates detections are: {prv_candidates}"
        )
        self.logger.debug(
            f"The oids for detections that are forced photometry or prv candidates and do not have the diaObjet field are:{dia_object}"
        )
        self.logger.info(
            "The number of features is: %i", len(model_input.features)
        )

    def predict_ztf(self, model_input):
        try:
            probabilities = self.model.predict_in_pipeline(
                model_input.features,
            )
            probabilities = OutputDTO(
                probabilities["probabilities"],
                probabilities["hierarchical"],
            )
            return probabilities
        except Exception as e:
            self.log_data(model_input)
            raise e

    def log_class_distribution(self, probabilities: OutputDTO):
        if not self.config.get("FEATURE_FLAGS", {}).get(
            "LOG_CLASS_DISTRIBUTION", False
        ):
            return
        df = probabilities.probabilities
        if "oid" in df.columns:
            df.set_index("oid", inplace=True)
        if "classifier_name" in df.columns:
            df = df.drop(["classifier_name"], axis=1)

        distribution = df.eq(df.where(df != 0).max(1), axis=0).astype(int)
        distribution = distribution.sum(axis=0)
        self.logger.debug("Class distribution:\n", distribution)

    def predict(self, model_input):
        if self.isztf:
            return self.predict_ztf(model_input)
        try:
            return self.model.predict(model_input)
        except ValueError as e:
            self.log_data(model_input)
            raise e

    def pre_execute(self, messages: List[dict]):
        if self.min_detections is None:
            return messages

        def has_enough_detections(message: dict) -> bool:
            n_dets = len(
                [True for det in message["detections"] if not det["forced"]]
            )
            return n_dets >= self.min_detections

        filtered_messages = filter(has_enough_detections, messages)
        filtered_messages = list(filtered_messages)
        return filtered_messages

    def execute(self, messages):
        """Run the classification.

        Parameters
        ----------
        messages : List[dict-like]
            Current object data, it must have the features and object id.

        """
        self.logger.info("Processing %i messages.", len(messages))
        model_input = create_input_dto(messages)
        probabilities = OutputDTO(
            DataFrame(), {"top": DataFrame(), "children": {}}
        )

        can, error = self.model.can_predict(model_input)
        if not can:
            self.logger.info(f"Can't predict\nError: {error}")
            return probabilities, messages, model_input.features

        probabilities = self.predict(model_input)
        self.log_class_distribution(probabilities)
        return probabilities, messages, model_input.features

    def post_execute(self, result: Tuple[OutputDTO, List[dict], DataFrame]):
        probabilities = result[0]
        parsed_result = self.scribe_parser.parse(
            probabilities,
            classifier_version=self.classifier_version,
        )
        self.produce_scribe(parsed_result.value)
        return result

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()

        else:
            self.consumer.__del__()
        self.producer.__del__()
