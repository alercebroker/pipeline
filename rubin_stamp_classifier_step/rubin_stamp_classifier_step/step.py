from typing import List, Union, Iterable, Dict, Any

from apf.consumers import KafkaConsumer
from apf.core.step import GenericStep
import logging
import numexpr
from .utils.tools import extract_image_from_fits
from .db.db import PSQLConnection, store_probability, get_taxonomy_by_classifier_id
from alerce_classifiers.base.dto import OutputDTO, InputDTO
from alerce_classifiers.base._types import (
    Detections,
    NonDetections,
    Features,
    Xmatch,
    Stamps,
)
from alerce_classifiers.rubin import StampClassifierModel
import pandas as pd



class StampClassifierStep(GenericStep):
    """
    Pipeline step for classifying stamps from LSST alerts.
    """

    def __init__(self, config: dict, level=logging.INFO, **step_args):
        super().__init__(config=config, level=level, **step_args)
        numexpr.utils.set_num_threads(1)
        self.model = StampClassifierModel(
            model_path=config["MODEL_CONFIG"]["MODEL_PATH"]
        )
        self.dict_mapping_classes = self.model.dict_mapping_classes
        self.psql_connection = PSQLConnection(config["DB_CONFIG"])
        if "CLS_ID" not in config["MODEL_CONFIG"]:
            self.classifier_id = 0
        else:
            self.classifier_id = config["MODEL_CONFIG"]["CLS_ID"]

        #aqui deberiamos obtener la taxonomia usando el classifier id
        #deberia haber una funcion en db.py con arg self.classifier_id

        self.class_taxonomy = get_taxonomy_by_classifier_id(self.classifier_id, self.psql_connection)
        logging.info(f"Class taxonomy: {self.class_taxonomy}")
    def pre_execute(self, messages: List[dict]) -> List[dict]:
        # Preprocessing: parsing, formatting and validation.

        # Extract required fields from messages
        # Metadata to provide:
        # ['airmass', 'magLim', 'psfFlux', 'psfFluxErr', 'scienceFlux',
        #  'scienceFluxErr', 'seeing', 'snr', 'ra', dec']

        # stamps: ['visit_image', 'difference_image', 'reference_image']

        logging.warning("Airmass is not available in schema v7.4, setting to 1.0")
        logging.warning("MagLim is not available in schema v7.4, setting to 25")
        logging.warning(
            "scienceFlux and scienceFluxErr are not available in schema v7.4, setting to 0"
        )
        logging.warning("Seeing is not available in schema v7.4, setting to 0.7")

        processed_messages = []
        #aqui deberia considerar todos los mensajes, si diaobject is none, entonces diasource no lo es
        for cont,message in enumerate(messages):
            #if message["diaObject"] is not None:
            processed_message = {}
            
            processed_message["diaObjectId"] = message["diaSource"]["diaObjectId"]
            processed_message["diaSourceId"] = message["diaSource"]["diaSourceId"]
            processed_message["ssObjectId"] = message["diaSource"]["ssObjectId"]

            obj_id = message["diaSource"]["diaObjectId"]
            src_id = message["diaSource"]["ssObjectId"]
            # Normalizamos: consideramos None o 0 como "falso"
            obj_ok = obj_id not in (None, 0)
            src_ok = src_id not in (None, 0)

            if obj_ok and src_ok:
                logging.info(
                    f"Both DiaObjectId and ssObjectId exists: obj_id={obj_id}, ss_id={src_id}"
                )
            elif not obj_ok and not src_ok:
                logging.info(
                    f"Both DiaObjectId and ssObjectId are None or 0"
                )

            # XOR lógico: solo pasa si uno es True y el otro False
            elif obj_ok ^ src_ok:

                processed_message["midpointMjdTai"] = message["diaSource"]["midpointMjdTai"]

                # Properties
                processed_message["ra"] = message["diaSource"]["ra"]
                processed_message["dec"] = message["diaSource"]["dec"]

                processed_message["airmass"] = 1.0
                processed_message["magLim"] = 25.0

                processed_message["psfFlux"] = message["diaSource"]["psfFlux"]
                processed_message["psfFluxErr"] = message["diaSource"]["psfFluxErr"]

                processed_message["scienceFlux"] = 0.0
                processed_message["scienceFluxErr"] = 0.0

                processed_message["seeing"] = 0.7

                processed_message["snr"] = message["diaSource"]["snr"]

                # Stamps
                processed_message["visit_image"] = extract_image_from_fits(
                    message["cutoutScience"]
                )
                processed_message["difference_image"] = extract_image_from_fits(
                    message["cutoutDifference"]
                )
                processed_message["reference_image"] = extract_image_from_fits(
                    message["cutoutTemplate"]
                )

                processed_messages.append(processed_message)

        return processed_messages

    def _messages_to_dto(self, messages: List[dict]) -> InputDTO:
        
        df = pd.DataFrame.from_records(messages)
        df = df.sort_values(by="midpointMjdTai").drop_duplicates(subset="diaObjectId", keep="first")

        df.set_index("diaObjectId", inplace=True)

        # Check if diaObjectId is unique
        if not df.index.is_unique:
            raise ValueError("diaObjectId must be unique in the input messages")

        # Create the InputDTO
        input_dto = InputDTO(
            Detections(pd.DataFrame()),
            NonDetections(pd.DataFrame()),
            Features(
                df[
                    [
                        "ra",
                        "dec",
                        "airmass",
                        "magLim",
                        "psfFlux",
                        "psfFluxErr",
                        "scienceFlux",
                        "scienceFluxErr",
                        "seeing",
                        "snr",
                    ]
                ]
            ),
            Xmatch(pd.DataFrame()),
            Stamps(
                df[
                    [
                        "visit_image",
                        "difference_image",
                        "reference_image",
                    ]
                ]
            ),
        )
        return input_dto

    def execute(
        self, messages: List[dict]
    ) -> Union[Iterable[Dict[str, Any]], Dict[str, Any]]:
        
        #aqui tengo que hacer la distincion si es diaobject none o no
        messages_to_process = [message for message in messages if message["diaObjectId"] is not None and message["diaObjectId"] != 0]
        messages_asteroids = [message for message in messages if message["ssObjectId"] is not None and message["ssObjectId"] != 0]
        if len(messages_to_process) > 0:
            input_dto = self._messages_to_dto(messages_to_process)
            output_dto: OutputDTO = self.model.predict(input_dto)
            predicted_probabilities = output_dto.probabilities
            #logging.info('input_dto:\n', input_dto)
            #logging.info('predicted_probabilities:\n', predicted_probabilities)

        #exit()

        output_messages = []
        for message in messages_to_process:
            output_message = {
                "diaObjectId": message["diaObjectId"],
                'ssObjectId': 0,
                "diaSourceId": message["diaSourceId"],
                "probabilities": predicted_probabilities.loc[
                    message["diaObjectId"]
                ].to_dict(),
                "midpointMjdTai": message["midpointMjdTai"],
                "ra": message["ra"],
                "dec": message["dec"],
            }
            output_messages.append(output_message)

        for message in messages_asteroids:
            output_message = {
                "diaObjectId": 0,
                'ssObjectId': message["ssObjectId"],
                "diaSourceId": message["diaSourceId"],
                "probabilities": {'AGN': 0.0, 
                                  'SN': 0.0, 
                                  'VS': 0.0, 
                                  'asteroid': 1.0, 
                                  'bogus': 0.0,},  # All probability to asteroid class
                "midpointMjdTai": message["midpointMjdTai"],
                "ra": message["ra"],
                "dec": message["dec"],
            }
            output_messages.append(output_message)

        return output_messages

    def post_execute(self, messages: List[dict]) -> List[dict]:
        #exit()
        # Write probabilities in the database
        store_probability(
            self.psql_connection,
            classifier_id=self.classifier_id,
            classifier_version=self.model.model_version,
            class_taxonomy = self.class_taxonomy,
            predictions=messages,
        )
        return messages

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
