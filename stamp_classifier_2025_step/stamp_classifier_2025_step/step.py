from apf.core import get_class
from apf.consumers import KafkaConsumer
from apf.core.step import GenericStep
from .db.db import PSQLConnection, store_probability
from .parsers.kafka_parser import KafkaParser
import logging
import numexpr
import traceback
import sys
from alerce_classifiers.base.dto import OutputDTO, InputDTO
from alerce_classifiers.base._types import *
from typing import List, Tuple
import pandas as pd
from pandas import DataFrame
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert


class MultiScaleStampClassifier(GenericStep):
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

        numexpr.utils.set_num_threads(1)
        
        """ CLASSSIFIER VERSION AND NAME"""
        self.classifier_name = self.config["MODEL_CONFIG"]["NAME"]
        self.classifier_version = self.config["MODEL_CONFIG"]["VERSION"]

        """ KAFKA PARSERS AND SCRIBE PARSERS """
        self.step_parser: KafkaParser = get_class(config["STEP_PARSER_CLASS"])()

        """ DB CONNECTION AND MODEL"""
        self.db_config = self.config["DATABASE_CREDENTIALS"]
        self.engine = PSQLConnection(self.db_config)
        self.mapper = get_class(config["MODEL_CONFIG"]["CLASS_MAPPER"])()
        self.model = get_class(config["MODEL_CONFIG"]["CLASS"])(
            **{"mapper": self.mapper, **config["MODEL_CONFIG"]["PARAMS"]}
        )

    def log_data(self, model_input):
        """BYPASS MODEL INPUT (NECESSARY ?)"""
        self.logger.info("data logger")

    def predict(self, model_input: InputDTO) -> OutputDTO | None:
        """MODELS ALREADY HAVE PREDICT METHOD (NECESSARY ??)"""
        try:
            return self.model.predict(model_input)
        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.print_exc())
            # sys.exit(1)

    def execute(self, messages):
        """Run the classification.

        Parameters
        ----------
        messages : List[dict-like]
            Current object data.

        """

        output_dto = OutputDTO(DataFrame(), {"top": DataFrame(), "children": {}})

        if len(messages) >= 1:
            input_dto = self._messages_to_input_dto(messages)
            self.logger.info(f"{len(messages)} consumed")
            try:
                self.logger.info(f"input : {input_dto}")
                output_dto = self.predict(input_dto)
                self.logger.info(f" output : {output_dto}")
                store_probability(self.engine, self.classifier_name, self.classifier_version, output_dto)
            except Exception as e:
                self.logger.error(e)
                self.logger.error(traceback.print_exc())

            return output_dto, messages, DataFrame()
        else:
            return output_dto, messages, DataFrame()
    

    def post_execute(self, result: Tuple[OutputDTO, List[dict], DataFrame]):

        return (
            result[0],
            result[1],
            DataFrame(),
        )
    
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
            Detections(DataFrame()),
            NonDetections(DataFrame()),
            Features(DataFrame()),
            Xmatch(DataFrame()),
            Stamps(
                pd.DataFrame.from_records(messages)
                .set_index("oid")
                .rename(columns=self._mapper_names)
            ),
        )

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
