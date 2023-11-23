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
        self.classifier_version = self.config["MODEL_VERSION"]
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

        self.scribe_producer = get_class(
            config["SCRIBE_PRODUCER_CONFIG"]["CLASS"]
        )(config["SCRIBE_PRODUCER_CONFIG"])
        self.scribe_parser: KafkaParser = get_class(
            config["SCRIBE_PARSER_CLASS"]
        )(classifier_name=self.classifier_name)
        self.step_parser: KafkaParser = get_class(
            config["STEP_PARSER_CLASS"]
        )()

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

    def execute(self, messages):
        """Run the classification.

        Parameters
        ----------
        messages : dict-like
            Current object data, it must have the features and object id.

        """
        self.logger.info("Processing %i messages.", len(messages))
        self.logger.debug("Messages received:\n", messages)
        self.logger.info("Getting batch alert data")
        model_input = create_input_dto(messages)
        self.oids = {}
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
            # oid hack for ztf
            if self.isztf:
                oids = self.oids.get(det[0], [])
                if det[1]["oid"] not in oids:
                    oids.append(det[1]["oid"])
                    self.oids[det[0]] = oids

        if not self.model.can_predict(model_input):
            self.logger.info("No data to process")
            return (
                OutputDTO(DataFrame(), {"top": DataFrame(), "children": {}}),
                messages,
                model_input.features,
            )
        self.logger.info(
            "The number of detections is: %i", len(model_input.detections)
        )
        self.logger.debug(f"The forced photometry detections are: {forced}")
        self.logger.debug(
            f"The prv candidates detections are: {prv_candidates}"
        )
        self.logger.debug(
            f"The aids for detections that are forced photometry or prv candidates and do not have the diaObjet field are:{dia_object}"
        )
        self.logger.info(
            "The number of features is: %i", len(model_input.features)
        )
        self.logger.info("Doing inference")
        if self.isztf:
            try:
                probabilities = self.model.predict_in_pipeline(
                    model_input.features,
                )
                probabilities = OutputDTO(
                    probabilities["probabilities"], probabilities["hierarchical"]
                )
            except Exception as e:
                print(e) # logger error?
                return (
                    OutputDTO(DataFrame(), {"top": DataFrame(), "children": {}}),
                    messages,
                    model_input.features,
                )
            
        else:
            try:
                probabilities = self.model.predict(model_input)
            except ValueError as e:
                self.logger.error(e)
                probabilities = OutputDTO(
                    DataFrame(), {"top": DataFrame(), "children": {}}
                )
        # after the former line, probabilities must be an OutputDTO

        self.logger.info("Processing results")
        df = probabilities.probabilities
        if "aid" in df.columns:
            df.set_index("aid", inplace=True)
        if "classifier_name" in df.columns:
            df = df.drop(["classifier_name"], axis=1)

        distribution = df.eq(df.where(df != 0).max(1), axis=0).astype(int)
        distribution = distribution.sum(axis=0)
        self.logger.debug("Class distribution:\n", distribution)
        return probabilities, messages, model_input.features

    def post_execute(self, result: Tuple[OutputDTO, List[dict]]):
        parsed_result = self.scribe_parser.parse(
            result[0],
            classifier_version=self.classifier_version,
            oids=self.oids,
        )
        self.produce_scribe(parsed_result.value)
        return result
