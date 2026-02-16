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

from xmatch_client import XmatchClient

from .database import (
    PSQLConnection,
    get_sql_references,
    get_feature_name_lut,
    get_or_create_version_id,
)

from .utils.metrics import get_sid
from .utils.parsers import parse_output, parse_scribe_payload
from .utils.parsers import detections_to_astro_object,detections_to_astro_object_lsst
from .utils.parsers import parse_output_lsst,parse_scribe_payload_lsst
from .utils.data_utils import clean_and_flatten_columns, save_astro_objects_to_csvs


from importlib.metadata import version


class FeatureStep(GenericStep):
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

        scribe_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = scribe_class(self.config["SCRIBE_PRODUCER_CONFIG"])

        self.scribe_topic_name = self.config["SCRIBE_PRODUCER_CONFIG"].get("TOPIC")
        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.FeatureStep")
        self.survey = self.config.get("SURVEY")
        self.use_xmatch = self.config.get("USE_XMATCH", False)
        
        # Get schema from configuration
        self.schema = self.config.get("DB_CONFIG", {}).get("SCHEMA", "multisurvey")

        if self.survey == "ztf":
            self.id_column = "candid"
            self.lightcurve_preprocessor = ZTFLightcurvePreprocessor(drop_bogus=True)
            self.feature_extractor = ZTFFeatureExtractor()
            self.extractor_group = ZTFFeatureExtractor.__name__
            self.detections_to_astro_object_fn = detections_to_astro_object
            self.parse_output_fn = parse_output
            self.parse_scribe_payload = parse_scribe_payload
            self.extractor_version = version("feature-step")
            self.feature_name_lut = None



        if self.survey == "lsst":
            self.id_column = "measurement_id"
            self.lightcurve_preprocessor = LSSTLightcurvePreprocessor()
            self.feature_extractor = LSSTFeatureExtractor()
            self.extractor_group = LSSTFeatureExtractor.__name__
            self.detections_to_astro_object_fn = detections_to_astro_object_lsst
            self.parse_output_fn = parse_output_lsst
            self.parse_scribe_payload = parse_scribe_payload_lsst
            
            # Get version name and resolve version_id from version_lut table
            version_name = version("feature-step")
            sid = 1
            tid = 1  
            self.extractor_version = get_or_create_version_id(
                self.db_sql, self.schema, version_name, sid, tid, self.logger
            )
            
            # Fetch feature name lookup table from multisurvey schema
            self.feature_name_lut = get_feature_name_lut(
                self.db_sql, self.schema, sid, tid, self.logger
            )

            # Initialize xmatch client
            if self.use_xmatch:
                xmatch_config = self.config.get("XMATCH_CONFIG", {})
                self.xmatch_client = XmatchClient(**xmatch_config)

        self.min_detections_features = config.get("MIN_DETECTIONS_FEATURES", None)
        if self.min_detections_features is None:
            self.min_detections_features = 1
        else:
            self.min_detections_features = int(self.min_detections_features)

    def get_xmatch_info(self, messages: List[dict]) -> List[Dict[str, Any]]:
        """
        Get xmatch information for LSST objects using conesearch.
        
        Parameters
        ----------
        messages : List[dict]
            List of messages containing oid, ra, dec information
            
        Returns
        -------
        List[Dict[str, Any]]
            List of xmatch results, each containing oid and match information
        """
        oids = []
        ras = []
        decs = []
        
        for msg in messages:
            if msg['sid'] == 2:
                continue  # Saltar sid 2
            oids.append(str(msg["oid"]))
            ras.append(msg["meanra"])
            decs.append(msg["meandec"])

        
        results = self.xmatch_client.conesearch_with_metadata(
            ras=ras, 
            decs=decs, 
            oids=oids
        )
        
        return results

    def produce_xmatch_to_scribe(self, xmatch_results: List[Dict[str, Any]], messages: List[dict]):
        """
        Produce xmatch results to scribe.
        
        Parameters
        ----------
        xmatch_results : List[Dict[str, Any]]
            List of xmatch results from conesearch_with_metadata
        messages : List[dict]
            Original messages to extract sid information
        """
        if not xmatch_results:
            return
        
        # Crear mapeo de oid -> sid desde los mensajes
        oid_to_sid = {}
        for msg in messages:            
            oid = str(msg['oid'])
            sid = msg['sid']
            oid_to_sid[oid] = sid
           
        
        # Procesar cada resultado de xmatch
        flush = False
        for idx, match in enumerate(xmatch_results):
            oid = str(match["oid"])
            sid = oid_to_sid.get(oid, 1)  # Usar sid del mensaje o valor por defecto
            
            # Crear comando en el formato especificado
            xmatch_command = {
                'step': 'xmatch',
                'survey': 'lsst',
                'payload': {
                    'oid': int(oid),
                    'sid': sid,
                    'catalog': match.get('catalog', 'allwise'),
                    'dist': match.get('distance'),
                    'oid_catalog': match.get('match_id')
                }
            }
            
            if idx == len(xmatch_results) - 1:
                flush = True
            
            self.scribe_producer.producer.produce(
                topic= self.scribe_topic_name,
                value=json.dumps(xmatch_command).encode("utf-8"),
                key=str(oid).encode("utf-8"),
            )

            if flush:
                self.scribe_producer.producer.flush()

    def produce_to_scribe(self, astro_objects: List[AstroObject]):
        commands = self.parse_scribe_payload(
            astro_objects,
            self.extractor_version,
            self.extractor_group,
            self.feature_name_lut
        )

        update_object_cmds = commands.get("update_object", [])
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

            if self.survey == "ztf":
                self.scribe_producer.produce({"payload": json.dumps(command)}, flush=flush)
            
            elif self.survey=="lsst":
                oid = command["payload"]["oid"]
                self.scribe_producer.producer.produce(
                    topic= self.scribe_topic_name,
                    value=json.dumps(command).encode("utf-8"),
                    key=str(oid).encode("utf-8"),
                )

                if flush:
                    self.scribe_producer.producer.flush()
            



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

        # Para LSST: obtener xmatch info para TODOS los mensajes antes de filtrar
        if self.use_xmatch and self.survey == "lsst" and len(messages) > 0:
            xmatch_results = self.get_xmatch_info(messages)
            # Crear diccionario de oid -> xmatch para búsqueda rápida
            xmatch_dict = {str(match['oid']): match for match in xmatch_results}
            
            for msg in messages:
                oid_str = str(msg['oid'])
                if oid_str in xmatch_dict:
                    msg['xmatches'] = xmatch_dict[oid_str]
                else:
                    msg['xmatches'] = None
            self.produce_xmatch_to_scribe(xmatch_results, messages)

        filtered_messages = []
        for message in messages:
            filtered_message = message.copy()
            if self.survey == "ztf":
                filtered_message["detections"] = discard_bogus_detections(
                    filtered_message.get("detections", [])
                )
                filtered_messages.append(filtered_message)
            elif self.survey == "lsst":
                dets = filtered_message.get('sources', []) + filtered_message.get('previous_sources', [])
                dets = [elem for elem in dets if elem.get('band') is not None]
                filtered_message['detections'] = dets
                filtered_messages.append(filtered_message)

        def has_enough_detections(message: dict) -> bool: # (ZTF)
            n_dets = len([True for det in message["detections"] if not det.get("forced", False)])
            return n_dets >= self.min_detections_features
        
        if self.survey == "ztf":
            filtered_messages = list(filter(has_enough_detections, filtered_messages))
        else:
            filtered_messages = list(filter(has_enough_detections, filtered_messages))

        if len(filtered_messages) > 0:
            self.logger.info("TIENE LENGTH MAYOR A CERO")

        return filtered_messages
    


    def execute(self, messages):

        candids = {}
        astro_objects = []
        messages_to_process = []

        oids = set()
        bands = set()
        for msg in messages:
            oids.add(msg["oid"])

        if self.survey == "ztf":
            references_db = self._get_sql_references(list(oids))

        for message in messages:
            if not message["oid"] in candids:
                candids[message["oid"]] = []
            candids[message["oid"]].extend(message[self.id_column]) #guarda los candid de cada oid
            m = map(
                lambda x: {**x, "index_column": str(x[self.id_column]) + "_" + str(x["oid"])},
                message.get("detections", []),
            )

            if self.survey == "ztf":
                xmatch_data = message["xmatches"]
                ao = self.detections_to_astro_object_fn(list(m), xmatch_data, references_db)
            else:
                forced = message.get("forced_sources", None) #si no hay detections, filtrar forced photometry
                xmatch_data = message.get("xmatches", None)
                ao = self.detections_to_astro_object_fn(list(m), forced, xmatch_data)
            astro_objects.append(ao)
            messages_to_process.append(message)

        self.lightcurve_preprocessor.preprocess_batch(astro_objects)
        self.feature_extractor.compute_features_batch(astro_objects, progress_bar=False)

        self.produce_to_scribe(astro_objects)
        output = self.parse_output_fn(astro_objects, messages_to_process, candids)
        return output

    def post_execute(self, result):
        
        self.metrics["sid"] = get_sid(result)

        for message in result:
            if "reference" in message:
                del message["reference"]

        return result

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
