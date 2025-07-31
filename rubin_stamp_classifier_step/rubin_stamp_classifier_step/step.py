from typing import List, Union, Iterable, Dict, Any

from apf.consumers import KafkaConsumer
from apf.core.step import GenericStep
import logging
import numexpr
from .utils.tools import extract_image_from_fits
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
        for message in messages:
            processed_message = {}

            # Object identity
            processed_message["alertId"] = message["alertId"]
            processed_message["diaObjectId"] = message["diaObject"]["diaObjectId"]
            processed_message["diaSourceId"] = message["diaSource"]["diaSourceId"]

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
        df.set_index("alertId", inplace=True)

        # Check if alertId is unique
        if not df.index.is_unique:
            raise ValueError("alertId must be unique in the input messages")

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
        input_dto = self._messages_to_dto(messages)
        output_dto = self.model.predict(input_dto)
        predicted_probabilities = output_dto.probabilities

        output_messages = []
        for message in messages:
            output_message = {
                "alertId": message["alertId"],
                "diaObjectId": message["diaObjectId"],
                "diaSourceId": message["diaSourceId"],
                "probabilities": predicted_probabilities.loc[
                    message["alertId"]
                ].to_dict(),
            }
            output_messages.append(output_message)
        return output_messages

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
