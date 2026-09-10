from typing import List, Union, Iterable, Dict, Any

from apf.consumers import KafkaConsumer
from apf.core import get_class
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
import json
from alerce_classifiers.rubin import StampClassifierModel
import pandas as pd



class StampClassifierStep(GenericStep):
    """
    Pipeline step for classifying stamps from LSST alerts.
    """

    DIA_OBJECT_SID = 1
    SS_OBJECT_SID = 2
    # Fields of schemas/rubin_stamp_classifier_step/output.avsc
    OUTPUT_FIELDS = (
        "diaObjectId",
        "ssObjectId",
        "diaSourceId",
        "probabilities",
        "midpointMjdTai",
        "ra",
        "dec",
    )

    def __init__(self, config: dict, level=logging.INFO, **step_args):
        super().__init__(config=config, level=level, **step_args)
        numexpr.utils.set_num_threads(1)
        self.model = StampClassifierModel(
            model_path=config["MODEL_CONFIG"]["MODEL_PATH"]
        )
        self.dict_mapping_classes = self.model.dict_mapping_classes
        self.psql_connection = PSQLConnection(config["DB_CONFIG"], poolclass="NullPool")
        self.survey = self.config.get("SURVEY")

        if "CLS_ID" not in config["MODEL_CONFIG"]:
            raise KeyError("MODEL_CONFIG.CLS_ID is required")
        self.classifier_id = config["MODEL_CONFIG"]["CLS_ID"]
        self.rename_stamp_columns = config.get("RENAME_STAMP_COLUMNS", False)

        self.class_taxonomy = get_taxonomy_by_classifier_id(self.classifier_id, self.psql_connection)
        logging.info(f"Class taxonomy: {self.class_taxonomy}")
        logging.info(f"RENAME_STAMP_COLUMNS: {self.rename_stamp_columns}")

        """ SCRIBE PRODUCER TO PRODUCE TO SCRIBE-MULTISURVEY TOPIC FOR ARCHIVAL PURPOSES"""
        scribe_cfg = config.get("SCRIBE_PRODUCER_CONFIG")
        self.scribe_producer = None
        self.scribe_topic_name = None
        if scribe_cfg:
            scribe_class = get_class(scribe_cfg["CLASS"])
            self.scribe_producer = scribe_class(scribe_cfg)
            self.scribe_topic_name = scribe_cfg.get("TOPIC")
            logging.info("Scribe producer enabled")
        else:
            logging.info("Scribe producer disabled (no config for scribe producer)")


    def pre_execute(self, messages: List[dict]) -> List[dict]:
        

        logging.warning("Airmass is not present in LSST alerts, setting to 1.0")
        logging.warning("MagLim is not present in LSST alerts, setting to 25")
        logging.warning("Seeing is not present in LSST alerts, setting to 0.7")

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
                # LSST identity, resolved once: diaObject -> sid 1, ssObject -> sid 2.
                # Everything downstream (model, db, scribe) reads oid and sid.
                if obj_ok:
                    processed_message["oid"] = obj_id
                    processed_message["sid"] = self.DIA_OBJECT_SID
                else:
                    processed_message["oid"] = src_id
                    processed_message["sid"] = self.SS_OBJECT_SID

                processed_message["midpointMjdTai"] = message["diaSource"]["midpointMjdTai"]

                # Properties
                processed_message["ra"] = message["diaSource"]["ra"]
                processed_message["dec"] = message["diaSource"]["dec"]

                processed_message["airmass"] = 1.0
                processed_message["magLim"] = 25.0

                processed_message["psfFlux"] = message["diaSource"]["psfFlux"]
                processed_message["psfFluxErr"] = message["diaSource"]["psfFluxErr"]

                processed_message["scienceFlux"] = message["diaSource"]["scienceFlux"]
                processed_message["scienceFluxErr"] = message["diaSource"]["scienceFluxErr"]

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
        df = df.sort_values(by="midpointMjdTai").drop_duplicates(subset="oid", keep="first")

        df.set_index("oid", inplace=True)

        if not df.index.is_unique:
            raise ValueError("oid must be unique in the input messages")

        stamps_df = df[
            [
                "visit_image",
                "difference_image",
                "reference_image",
            ]
        ]
        if self.rename_stamp_columns:
            stamps_df = stamps_df.rename(
                columns={
                    "visit_image": "flux_Science_data",
                    "difference_image": "flux_Difference_data",
                    "reference_image": "flux_Template_data",
                }
            )

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
                        "sid",
                    ]
                ]
            ),
            Xmatch(pd.DataFrame()),
            Stamps(stamps_df),
        )
        return input_dto

    def execute(
        self, messages: List[dict]
    ) -> Union[Iterable[Dict[str, Any]], Dict[str, Any]]:
        
        # Every alert goes to the model; what it does with sid (e.g. the
        # asteroid rule for solar system objects) is the model's business.
        input_dto = self._messages_to_dto(messages)
        output_dto: OutputDTO = self.model.predict(input_dto)
        predicted_probabilities = output_dto.probabilities

        output_messages = []
        for message in messages:
            oid, sid = message["oid"], message["sid"]
            output_messages.append(
                {
                    "oid": oid,
                    "sid": sid,
                    "diaObjectId": oid if sid == self.DIA_OBJECT_SID else 0,
                    "ssObjectId": oid if sid == self.SS_OBJECT_SID else 0,
                    "diaSourceId": message["diaSourceId"],
                    "probabilities": predicted_probabilities.loc[oid].to_dict(),
                    "midpointMjdTai": message["midpointMjdTai"],
                    "ra": message["ra"],
                    "dec": message["dec"],
                }
            )

        return output_messages

    def post_execute(self, messages: List[dict]) -> List[dict]:

        # Write probabilities in the database
        store_probability(
            self.psql_connection,
            classifier_id=self.classifier_id,
            classifier_version=self.model.model_version,
            class_taxonomy = self.class_taxonomy,
            predictions=messages,
        )

        # Produce to scribe
        if self.scribe_producer is not None:
            self.produce_to_scribe(messages)

        return messages

    def pre_produce(self, messages: List[dict]) -> List[dict]:
        # oid and sid are internal; the output schema is strict about extra fields.
        return [
            {key: value for key, value in message.items() if key in self.OUTPUT_FIELDS}
            for message in messages
        ]

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
    

    def _format_scribe_records(self, predictions: list[dict]) -> list[dict]:
        records = []

        for msg in predictions:
            probs = msg["probabilities"]

            # format the message for all ranks
            sorted_classes = sorted(probs.items(), key=lambda x: x[1], reverse=True)

            for rank, (class_name, prob) in enumerate(sorted_classes, start=1):
                records.append(
                    {
                        "oid": msg["oid"],
                        "sid": msg["sid"],
                        "classifier_id": self.classifier_id,
                        "classifier_version": int(self.model.model_version.replace(".", "")),
                        "class_id": self.class_taxonomy.get(class_name, -1),
                        "probability": prob,
                        "ranking": rank,
                        "lastmjd": msg["midpointMjdTai"],
                    }
                )
                

        return records
    
    def produce_to_scribe(self, predictions: list[dict]):

        # When no scribe producer is configured, do nothing
        if self.scribe_producer is None:
            return
        
        records = self._format_scribe_records(predictions)
        if not records:
            return

        last = len(records) - 1

        for i, record in enumerate(records):
            command = {
                "step": "probability-archival-step",
                "survey": "lsst",
                "payload": record,
            }

            self.scribe_producer.producer.produce(
                topic=self.scribe_topic_name,
                value=json.dumps(command).encode("utf-8"),
                key=str(record["oid"]).encode("utf-8"),
            )

            if i == last:
                self.scribe_producer.producer.flush()