# Standard library
import logging
import sys
import traceback
import io
import gzip
from typing import List, Tuple

# Third-party
import numexpr
import numpy as np
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from astropy.io import fits

# Alerce and APF
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
from alerce_classifiers.base._types import (
    Detections,
    NonDetections,
    Features,
    Xmatch,
    Stamps,
)
from db_plugins.db.sql.models import Probability


class MultiScaleStampClassifier(GenericStep):
    """
    Classifier step for processing multi-scale stamps.
    """

    _mapper_names = {
        "cutoutScience": "science",
        "cutoutTemplate": "reference",
        "cutoutDifference": "diff",
    }

    _mapper_isdiffpos = {
        "f": -1,
        "t": 1,
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

    def _read_and_transform_messages(self, messages: list[dict]) -> pd.DataFrame:
        """read compressed messages and return lightweight messages with only necesary data"""
        records = []
        for msg in messages:
            try:
                science = self._decode_fits(msg["cutoutScience"]["stampData"])
                if science.shape == (63, 63):
                    template = self._decode_fits(msg["cutoutTemplate"]["stampData"])
                    diff = self._decode_fits(msg["cutoutDifference"]["stampData"])
                    metadata = self._extract_metadata_from_message(msg)
                    metadata["cutoutScience"] = science
                    metadata["cutoutTemplate"] = template
                    metadata["cutoutDifference"] = diff
                    records.append(metadata)
            except Exception as e:
                logging.warning(f"Skipping message due to error: {e}")

        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df = df.set_index("oid").sort_values("jd", ascending=True)
            df = df[~df.index.duplicated(keep="first")]
            messages_dict = df.to_dict(orient="index")
        else:
            messages_dict = {}

        logging.info(f"Kept {len(records)}/{len(messages)} messages with 63x63 stamps.")
        return df, messages_dict

        # USANDO PADDING DIRECTO
        # records = []
        # for msg in messages:
        #    template = self._extract_metadata_from_message(msg)
        #    for k in ["cutoutScience", "cutoutTemplate", "cutoutDifference"]:
        #        template[k] = self._pad_matrices(
        #            matrix=self._decode_fits(msg[k]["stampData"]),
        #            target_shape=(63, 63),
        #        )
        #    records.append(template)

    #
    # df = pd.DataFrame.from_records(records).set_index('oid').sort_values("jd", ascending=True)
    # df = df[~df.index.duplicated(keep="first")]
    # messages_dict = df.to_dict(orient="index")
    # return df, messages_dict

    def _df_to_input_dto(self, df: pd.DataFrame) -> InputDTO:
        return InputDTO(
            Detections(DataFrame()),
            NonDetections(DataFrame()),
            Features(DataFrame()),
            Xmatch(DataFrame()),
            Stamps(df.rename(columns=self._mapper_names)),
        )

    def predict(self, model_input: InputDTO) -> OutputDTO | None:
        """MODELS ALREADY HAVE PREDICT METHOD (NECESSARY ??)"""
        try:
            return self.model.predict(model_input)
        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.print_exc())

    def _extract_metadata_from_message(self, msg: dict) -> dict:
        return {
            "oid": msg["objectId"],
            "jd": msg["candidate"]["jd"],
            "ra": msg["candidate"]["ra"],
            "dec": msg["candidate"]["dec"],
            "ssdistnr": msg["candidate"]["ssdistnr"],
            "isdiffpos": self._mapper_isdiffpos[msg["candidate"]["isdiffpos"]],
            "sgscore1": msg["candidate"]["sgscore1"],
            "distpsnr1": msg["candidate"]["distpsnr1"],
            "candid": msg["candidate"]["candid"],
        }

    def _pad_matrices(self, matrix: np.ndarray, target_shape: tuple) -> np.ndarray:
        current_shape = matrix.shape
        if current_shape[0] > target_shape[0] or current_shape[1] > target_shape[1]:
            raise ValueError(
                "Target shape must be greater than or equal to the current shape."
            )
        pad_rows = target_shape[0] - current_shape[0]
        pad_cols = target_shape[1] - current_shape[1]
        return np.pad(
            matrix, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0
        )

    def _decode_fits(self, data: bytes) -> np.array:
        import io
        import gzip
        from astropy.io import fits

        with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
            decompressed_data = f.read()
            with fits.open(io.BytesIO(decompressed_data)) as hdul:
                return hdul[0].data

    def _check_dimension_stamps(self, messages: list[dict]) -> List[dict]:
        """ensure that all stamps share the same dimension"""
        messages_filtered = []
        for msg in messages:
            condition_01 = (
                msg["data_stamp_inference"]["cutoutScience"].shape
                == msg["data_stamp_inference"]["cutoutTemplate"].shape
            )
            condition_02 = (
                msg["data_stamp_inference"]["cutoutScience"].shape
                == msg["data_stamp_inference"]["cutoutDifference"].shape
            )
            if condition_01 and condition_02:
                self.logger.info(
                    f"stamps with the same dimension dim = {msg['data_stamp_inference']['cutoutScience'].shape}"
                )
                messages_filtered.append(msg)

        return messages_filtered

    def _messages_to_input_dto(self, messages: List[dict]) -> InputDTO:
        ## get only necessary information form each msg
        records = []
        for msg in messages:
            records.append(msg["data_stamp_inference"])

        return InputDTO(
            Detections(DataFrame()),
            NonDetections(DataFrame()),
            Features(DataFrame()),
            Xmatch(DataFrame()),
            Stamps(
                pd.DataFrame.from_records(records)
                .set_index("oid")
                .rename(columns=self._mapper_names)
            ),
        )

    def execute(self, messages):
        """Run the classification.

        Parameters
        ----------
        messages : List[dict-like]
            Current object data.

        """

        output_dto = OutputDTO(DataFrame(), {"top": DataFrame(), "children": {}})

        if len(messages) >= 1:
            df_stamps, messages_dict = self._read_and_transform_messages(messages)

            try:
                input_dto = self._df_to_input_dto(df_stamps)
                self.logger.info(f"{len(messages_dict)} messages consumed")
            except Exception as e:
                self.logger.error("Error converting DataFrame to InputDTO")
                self.logger.error(e)
                self.logger.error(traceback.format_exc())
                return output_dto, messages_dict, DataFrame()

            self.logger.info(f"input : {input_dto}")
            output_dto = self.predict(input_dto)
            self.logger.info(f" output : {output_dto}")
            store_probability(
                self.engine, self.classifier_name, self.classifier_version, output_dto
            )

            oids_removed = [
                k
                for k in messages_dict.keys()
                if k not in output_dto.probabilities.index
            ]
            n_messages_total = len(messages_dict)
            for oid in oids_removed:
                messages_dict.pop(oid)
            logging.info(
                f"Kept {len(messages_dict)}/{n_messages_total} messages with valid probabilities."
            )

            return output_dto, messages_dict, DataFrame()
        else:
            return output_dto, messages, DataFrame()

    def post_execute(self, result: Tuple[OutputDTO, List[dict], DataFrame]):
        return (
            result[0],
            result[1],
            DataFrame(),
        )

    def pre_produce(self, result: Tuple[OutputDTO, List[dict], DataFrame]):
        return self.step_parser.parse(
            model_output=result[0],
            messages=result[1],
        ).value

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()