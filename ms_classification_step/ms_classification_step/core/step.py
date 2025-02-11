from apf.core import get_class
from apf.consumers import KafkaConsumer
from apf.core.step import GenericStep
from lc_classification.core.parsers.kafka_parser import KafkaParser
import logging
import numexpr
from alerce_classifiers.base.dto import OutputDTO, InputDTO
from typing import List, Tuple
import pandas as pd
from pandas import DataFrame
import numpy as np

class MultiStampClassifier(GenericStep):
    """
    MultiScaleStampClassifier
    """

    _mapper_names = {
        "cutoutScience": "science",
        "cutoutTemplate": "reference",
        "cutoutDifference": "diff",
    }

    def __init__(self, config={}, level=logging.INFO, model=None, **step_args):
        super().__init__(config=config, level=level, **step_args)
        # self.classifier_name = self.config["MODEL_CONFIG"]["NAME"]
        numexpr.utils.set_num_threads(1)

        # self.isztf = config["MODEL_CONFIG"]["CLASS"] == ZTF_CLASSIFIER_CLASS
        self.logger.info("Loading Models")

        # self.scribe_producer = get_class(
        #     config["SCRIBE_PRODUCER_CONFIG"]["CLASS"]
        # )(config["SCRIBE_PRODUCER_CONFIG"])
        # self.scribe_parser: KafkaParser = get_class(
        #     config["SCRIBE_PARSER_CLASS"]
        # )(classifier_name=self.classifier_name)
        self.step_parser: KafkaParser = get_class(config["STEP_PARSER_CLASS"])()
        self.model = get_class(config["MODEL_CONFIG"]["CLASS"])(
            **config["MODEL_CONFIG"]["PARAMS"]
        )


    def pre_produce(self, result: Tuple[OutputDTO, List[dict], DataFrame]):
        pass
        # return self.step_parser.parse(
        #     model_output=OutputDTO(DataFrame(), {"top": DataFrame(), "children": {}}),
        #     messages=[],
        #     features=DataFrame(),
        #     # result[0],
        #     # messages=result[1],
        #     # features=result[2],
        #     # classifier_name=self.classifier_name,
        #     # classifier_version=self.classifier_version,
        # )

    def produce_scribe(self, commands: List[dict]):
        pass
        # ids_list = []
        # for command in commands:
        #     ids_list.append(command["criteria"]["_id"])
        #     self.scribe_producer.produce({"payload": json.dumps(command)})
        # self.logger.debug(f"The list of objets from scribe are: {ids_list}")

    def produce(self, result):
        """avoid produccing messages"""
        pass

    def log_data(self, model_input):
        self.logger.info("data logger")

    def predict(self, model_input):
        return self.model.predict(model_input)

    def execute(self, messages):
        """Run the classification.

        Parameters
        ----------
        messages : List[dict-like]
            Current object data.

        """
        input_dto = self._messages_to_input_dto(messages)
        probabilities = OutputDTO(DataFrame(), {"top": DataFrame(), "children": {}})
        return probabilities, [], DataFrame()

    def post_execute(self, result: Tuple[OutputDTO, List[dict], DataFrame]):

        return (
            OutputDTO(DataFrame(), {"top": DataFrame(), "children": {}}),
            [{}],
            DataFrame(),
        )

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()

    def pre_execute(self, messages):
        """override method"""
        return self._read_and_transform_messages(messages)

    def _read_and_transform_messages(self, messages) -> List[dict]:
        """read compressed messages and return lightweight messages with only necesary data"""
        for i, msg in enumerate(messages):
            """for each message extract only necessary information"""
            template = {}
            template.update(self._extract_metadata_from_message(msg))
            for k in ["cutoutScience", "cutoutTemplate", "cutoutDifference"]:
                template.update({k: self._decode_fits(msg[k]["stampData"])})
            """ update the same list """
            messages[i] = template
        return messages

    def _extract_metadata_from_message(self, msg: dict) -> dict:

        return {
            "oid": msg["objectId"], 
            "ra": msg["candidate"]["ra"],
            "dec": msg["candidate"]["dec"],
            "candid": msg["candidate"]["candid"],
        }

    def _decode_fits(self, data: bytes) -> np.array:

        import io
        import gzip
        from astropy.io import fits

        with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
            decompressed_data = f.read()
            with fits.open(io.BytesIO(decompressed_data)) as hdul:
                return hdul[0].data

    def _messages_to_input_dto(self, messages: List[dict]) -> InputDTO:

        return InputDTO(
            DataFrame,
            DataFrame,
            DataFrame,
            DataFrame,
            pd.DataFrame.from_records(messages)
            .set_index("oid")
            .rename(columns=self._mapper_names),
        )
