import pandas as pd
import logging
import json
from typing import Any, Dict, Iterable, List, Optional

from apf.core import get_class
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer

from lc_classifier.features.core.base import AstroObject, discard_bogus_detections
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor #me falta crear esto
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor #me falta crear esto
from lc_classifier.features.composites.lsst import LSSTFeatureExtractor 
from lc_classifier.features.preprocess.lsst import LSSTLightcurvePreprocessor


from .database import (
    PSQLConnection,
    get_sql_references,
)

from .utils.metrics import get_sid
from .utils.parsers import parse_output, parse_scribe_payload
from .utils.parsers import detections_to_astro_object,detections_to_astro_object_lsst

from importlib.metadata import version


class FeatureStep(GenericStep): #qua la saque del environment
    """FeatureStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(
        self,
        config=None,
        db_sql: PSQLConnection = None,
        **step_args,
    ):

        super().__init__(config=config, **step_args)
        # Bogus detections are dropped in pre_execute
       

        scribe_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = scribe_class(self.config["SCRIBE_PRODUCER_CONFIG"])
        self.extractor_version = version("feature-step")
        self.extractor_group = ZTFFeatureExtractor.__name__

        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.FeatureStep")
        self.survey = "LSST"#config.get("SURVEY", "ZTF")

        if self.survey == "ZTF":
            self.lightcurve_preprocessor = ZTFLightcurvePreprocessor(drop_bogus=True)
            self.feature_extractor = ZTFFeatureExtractor()
        if self.survey == "LSST":
            self.lightcurve_preprocessor = LSSTLightcurvePreprocessor()
            self.feature_extractor = LSSTFeatureExtractor()
        self.min_detections_features = config.get("MIN_DETECTIONS_FEATURES", None)
        if self.min_detections_features is None:
            self.min_detections_features = 1
        else:
            self.min_detections_features = int(self.min_detections_features)

    def produce_to_scribe(self, astro_objects: List[AstroObject]):
        commands = parse_scribe_payload(
            astro_objects,
            self.extractor_version,
            self.extractor_group,
        )
        update_object_cmds = commands["update_object"]
        update_features_cmds = commands["upserting_features"]

        count_objs = 0
        flush = False
        for command in update_object_cmds:
            count_objs += 1
            if count_objs == len(update_object_cmds):
                flush = True
            self.scribe_producer.produce({"payload": json.dumps(command)}, flush=flush)

        count_features = 0
        flush = False
        for command in update_features_cmds:
            count_features += 1
            if count_features == len(update_features_cmds):
                flush = True
            self.scribe_producer.produce({"payload": json.dumps(command)}, flush=flush)

    def pre_produce(self, result: Iterable[Dict[str, Any]] | Dict[str, Any]):
        self.set_producer_key_field("oid")
        return result

    def _get_sql_references(self, oids: List[str]) -> Optional[pd.DataFrame]:
        db_references = get_sql_references(
            oids, self.db_sql, keys=["oid", "rfid", "sharpnr", "chinr"]
        )
        db_references = db_references[db_references["chinr"] >= 0.0].copy()
        return db_references

    def pre_execute(self, messages: List[dict]):
        filtered_messages = []

        for message in messages:
            filtered_message = message.copy()
            #me falta implementar el discard_bogus_detections para lsst
            if self.survey == "ZTF":
                filtered_message["detections"] = discard_bogus_detections(
                    filtered_message["detections"]
                )
        filtered_messages.append(filtered_message)

        def has_enough_detections(message: dict) -> bool: #esto va a fallar, por no tener el campo forced
            n_dets = len([True for det in message["detections"] if not det["forced"]])
            return n_dets >= self.min_detections_features
        
        def has_enough_detections_lsst(message: dict) -> bool: #esto va a fallar, por no tener el campo forced
            n_dets = len([True for det in message["detections"]])
            return True#n_dets >= self.min_detections_features

        if self.survey == "ZTF":
            filtered_messages = filter(has_enough_detections, filtered_messages)
        if self.survey == "LSST":
            filtered_messages = filter(has_enough_detections_lsst, filtered_messages)
        filtered_messages = list(filtered_messages)
        return filtered_messages

    def execute(self, messages):
        candids = {}
        astro_objects = []
        messages_to_process = []

        oids = set()
        for msg in messages:
            oids.add(msg["oid"])

        if self.survey == "ZTF":
            references_db = self._get_sql_references(list(oids))

        if self.survey == "ZTF":
            for message in messages: #que hace aqui? para lsst, en vez de candid, ocupare diapbjectId
                if not message["oid"] in candids:
                    candids[message["oid"]] = []
                candids[message["oid"]].extend(message["candid"]) #guarda los candid de cada oid
                m = map(
                    lambda x: {**x, "index_column": str(x["candid"]) + "_" + x["oid"]},
                    message.get("detections", []),
                )
                xmatch_data = message["xmatches"]

                ao = detections_to_astro_object(list(m), xmatch_data, references_db)
                astro_objects.append(ao)
                messages_to_process.append(message)

        if self.survey == "LSST":
            for message in messages: #que hace aqui? para lsst, en vez de candid, ocupare diapbjectId
                if not message["oid"] in candids:
                    candids[message["oid"]] = []
                candids[message["oid"]].extend(message["diaObjectId"]) #guarda los candid de cada oid
                m = map(
                    lambda x: {**x, "index_column": str(x["diaObjectId"]) + "_" + x["oid"]},
                    message.get("detections", []),
                )
                #xmatch_data = message["xmatches"]
                forced = message.get("forced", False) #si no hay detections, filtrar forced photometry
                #me falta incluir la photometria forzada en el input

                ao = detections_to_astro_object_lsst(list(m), forced)
                astro_objects.append(ao)
                messages_to_process.append(message)

        self.lightcurve_preprocessor.preprocess_batch(astro_objects)
        self.feature_extractor.compute_features_batch(astro_objects, progress_bar=False)

        print("Llegue aqui")

        #me falta la parsear la salida
        self.produce_to_scribe(astro_objects)
        output = parse_output(astro_objects, messages_to_process, candids)
        return output

    def post_execute(self, result):
        self.metrics["sid"] = get_sid(result)

        for message in result:
            del message["reference"]

        return result

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
