from apf.core import get_class
from apf.consumers import KafkaConsumer
from apf.core.step import GenericStep
from db_plugins.db.sql.models import Probability
from ms_classification_step.core.parsers.kafka_parser import KafkaParser
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
        # TODO:
        # self.scribe_producer = get_class(
        #     config["SCRIBE_PRODUCER_CONFIG"]["CLASS"]
        # )(config["SCRIBE_PRODUCER_CONFIG"])
        # self.scribe_parser: KafkaParser = get_class(
        #     config["SCRIBE_PARSER_CLASS"]
        # )(classifier_name=self.classifier_name)

        """ DB CONNECTION AND MODEL"""
        self.engine = self._create_engie()
        self.mapper = get_class(config["MODEL_CONFIG"]["CLASS_MAPPER"])()
        self.model = get_class(config["MODEL_CONFIG"]["CLASS"])(
            **{"mapper": self.mapper, **config["MODEL_CONFIG"]["PARAMS"]}
        )

    def pre_produce(self, result: Tuple[OutputDTO, List[dict], DataFrame]):
        return self.step_parser.parse(
            model_output=result[0],
            messages=result[1],
        ).value

    def produce_scribe(self, commands: List[dict]):
        """BYPASS FUNCTION"""
        pass
        # ids_list = []
        # for command in commands:
        #     ids_list.append(command["criteria"]["_id"])
        #     self.scribe_producer.produce({"payload": json.dumps(command)})
        # self.logger.debug(f"The list of objets from scribe are: {ids_list}")

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
                self._insert_to_db_from_output_dto(output_dto)
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

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()

    def pre_execute(self, messages):
        """override method"""
        return self._read_and_transform_messages(messages)

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

    def _read_and_transform_messages(self, messages: list[dict]) -> List[dict]:
        """read compressed messages and return lightweight messages with only necesary data"""
        for i, msg in enumerate(messages):
            """for each message extract only necessary information"""
            template = {}
            template.update(self._extract_metadata_from_message(msg))
            for k in ["cutoutScience", "cutoutTemplate", "cutoutDifference"]:
                template.update(
                    {
                        k: self._pad_matrices(
                            matrix=self._decode_fits(msg[k]["stampData"]),
                            target_shape=(63, 63),
                        )
                    }
                )
            """ update the same list """
            messages[i].update({"data_stamp_inference": template})
        return messages

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

    def _get_ranking(self, output_dto: OutputDTO) -> DataFrame:
        probabilitites = output_dto.probabilities.reset_index()
        probabilitites_melt = probabilitites.melt(
            id_vars=["oid"], var_name="class_name", value_name="probability"
        )
        probabilitites_melt["ranking"] = probabilitites_melt.groupby("oid")[
            "probability"
        ].rank(ascending=False, method="dense")
        return probabilitites_melt

    def _insert_to_db_from_output_dto(self, output_dto: OutputDTO) -> None:
        probabilities = self._get_ranking(output_dto)
        probabilities["classifier_name"] = self.classifier_name
        probabilities["classifier_version"] = self.classifier_version
        """ to records """
        probabilities = probabilities.to_dict(orient="records")
        """ probabilities insert to database """
        try:
            insert_stmt = insert(Probability).values(probabilities)
            insert_stmt = insert_stmt.on_conflict_do_nothing()
            with self.model_psql.connect() as conn:
                conn.execute(insert_stmt)
                conn.commit()

        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.print_exc())

    def _create_engie(self):
        try:
            self.db_config = self.config["DATABASE_CREDENTIALS"]
            self.model_psql = create_engine(
                f"postgresql://{self.db_config['USER']}:{self.db_config['PASSWORD']}@{self.db_config['HOST']}:{self.db_config['PORT']}/{self.db_config['DB_NAME']}",
                connect_args={
                    "options": "-csearch_path={}".format(self.db_config["SCHEMA"])
                },
            )
        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.print_exc())
            sys.exit(1)
