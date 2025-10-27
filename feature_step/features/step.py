import pandas as pd
import logging
import json
import os
import uuid
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
from .utils.parsers import parse_output_lsst,parse_scribe_payload_lsst


from importlib.metadata import version


def clean_and_flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana columnas con listas y remueve saltos de línea en strings."""
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df = df.explode(col)
        df[col] = df[col].apply(lambda x: str(x).replace('\n', ' ') if isinstance(x, str) else x)
    return df


def save_astro_objects_to_csvs(
    astro_objects: List["AstroObject"],
    messages_to_process: List[dict],
    base_folder: str = "csvs",
) -> str:
    """Guarda detections y features de cada AstroObject en CSVs por OID.

    Crea una carpeta base si no existe y un subfolder por batch con UUID.
    Retorna la ruta del folder del batch.
    """
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    batch_id = str(uuid.uuid4())
    batch_folder = os.path.join(base_folder, batch_id)
    os.makedirs(batch_folder)
    #print(len(astro_objects))

    for i, (ao, msg) in enumerate(zip(astro_objects, messages_to_process)):
        oid = getattr(ao, "oid", msg.get("oid", f"obj_{i}"))
        #print(oid)
        detections_csv_path = os.path.join(batch_folder, f"{oid}_detections.csv")
        features_csv_path = os.path.join(batch_folder, f"{oid}_features.csv")

        detections_df = clean_and_flatten_columns(ao.detections)
        features_df = clean_and_flatten_columns(ao.features)

        if "sid" in detections_df.columns:
            detections_df = detections_df.drop(columns=["sid"])
        if "sid" in features_df.columns:
            features_df = features_df.drop(columns=["sid"])

        detections_df.to_csv(detections_csv_path, index=False)
        features_df.to_csv(features_csv_path, index=False)

        print(f"Saved: {detections_csv_path}")
        print(f"Saved: {features_csv_path}")

    return batch_folder


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

        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.FeatureStep")
        self.survey = "LSST"#config.get("SURVEY", "ZTF") #tengo que llevar esto a archivo de config

        if self.survey == "ZTF":
            self.id_column = "candid"
            self.lightcurve_preprocessor = ZTFLightcurvePreprocessor(drop_bogus=True)
            self.feature_extractor = ZTFFeatureExtractor()
            self.extractor_group = ZTFFeatureExtractor.__name__
            self.detections_to_astro_object_fn = detections_to_astro_object
            self.parse_output_fn = parse_output
            self.parse_scribe_payload = parse_scribe_payload
            self.extractor_version = version("feature-step")


        if self.survey == "LSST":
            self.id_column = "measurement_id"
            self.lightcurve_preprocessor = LSSTLightcurvePreprocessor()
            self.feature_extractor = LSSTFeatureExtractor()
            self.extractor_group = LSSTFeatureExtractor.__name__
            self.detections_to_astro_object_fn = detections_to_astro_object_lsst
            self.parse_output_fn = parse_output_lsst
            self.parse_scribe_payload = parse_scribe_payload_lsst
            self.extractor_version = version("feature-step")



        self.min_detections_features = config.get("MIN_DETECTIONS_FEATURES", None)
        if self.min_detections_features is None:
            self.min_detections_features = 1
        else:
            self.min_detections_features = int(self.min_detections_features)

    def produce_to_scribe(self, astro_objects: List[AstroObject]):
        commands = self.parse_scribe_payload(
            astro_objects,
            self.extractor_version,
            self.extractor_group,
        ) #llegar hasta aqui es facil. Correr ZTF y correr LSST
        #print(commands)
        update_object_cmds = commands.get("update_object", [])
        update_features_cmds = commands["payload"]

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

    def _get_sql_references(self, oids: List[str]) -> Optional[pd.DataFrame]: #esto es solo ZTF
        db_references = get_sql_references(
            oids, self.db_sql, keys=["oid", "rfid", "sharpnr", "chinr"]
        )
        db_references = db_references[db_references["chinr"] >= 0.0].copy()
        return db_references

    def pre_execute(self, messages: List[dict]):
        # Guardar en JSON los messages que cumplan que len(sources)+len(previous_sources) > 10
        """try:
            msgs_to_save = [
                m for m in messages
                if (len(m.get("sources", [])) + len(m.get("previous_sources", []))) > 10
            ]
            if msgs_to_save:
                jsons_folder = "jsons"
                os.makedirs(jsons_folder, exist_ok=True)
                file_id = str(uuid.uuid4())
                json_path = os.path.join(jsons_folder, f"messages_{file_id}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(msgs_to_save, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Saved filtered messages JSON: {json_path} ({len(msgs_to_save)} items)")
        except Exception as e:
            self.logger.exception(f"Failed to save filtered messages JSON: {e}")"""

        filtered_messages = []
        #print(messages[0])

        for message in messages:
            filtered_message = message.copy()
            if self.survey == "ZTF":
                filtered_message["detections"] = discard_bogus_detections(
                    filtered_message.get("detections", [])
                )
                filtered_messages.append(filtered_message)
            elif self.survey == "LSST":
                # Solo conservar mensajes con >10 (sources + previous_sources)
                #total_src = len(filtered_message.get('sources', [])) + len(filtered_message.get('previous_sources', []))
                #if total_src <= 10:
                #    continue
                dets = filtered_message.get('sources', []) + filtered_message.get('previous_sources', [])
                dets = [elem for elem in dets if elem.get('band') is not None]
                filtered_message['detections'] = dets
                filtered_messages.append(filtered_message)

        def has_enough_detections(message: dict) -> bool: # (ZTF)
            n_dets = len([True for det in message["detections"] if not det.get("forced", False)])
            return n_dets >= self.min_detections_features
        
        # En LSST ya filtramos por >10 sources+previous_sources, no aplicar filtro adicional
        if self.survey == "ZTF":
            filtered_messages = list(filter(has_enough_detections, filtered_messages))
        else:
            filtered_messages = list(filter(has_enough_detections, filtered_messages))
        return filtered_messages

    def execute(self, messages):
        candids = {}
        astro_objects = []
        messages_to_process = []

        oids = set()
        bands = set()
        for msg in messages:
            oids.add(msg["oid"])

        if self.survey == "ZTF":
            references_db = self._get_sql_references(list(oids))
        for message in messages:
            if not message["oid"] in candids:
                candids[message["oid"]] = []
            candids[message["oid"]].extend(message[self.id_column]) #guarda los candid de cada oid
            m = map(
                lambda x: {**x, "index_column": str(x[self.id_column]) + "_" + str(x["oid"])},
                message.get("detections", []),
            )

            if self.survey == "ZTF":
                xmatch_data = message["xmatches"]
                ao = self.detections_to_astro_object_fn(list(m), xmatch_data,references_db)
            else:
                forced = message.get("forced", False) #si no hay detections, filtrar forced photometry
                #print(len(list(m)))
                ao = self.detections_to_astro_object_fn(list(m), forced)
                #print('FID:',ao.detections['fid'].unique())
                #print(list(m))
                #for source in message['detections']:
                #    bands.add(source['band'])
                #    if source['band'] == 2:
                #        print(msg["oid"])

            astro_objects.append(ao)
            messages_to_process.append(message)
        #print(bands)

        self.lightcurve_preprocessor.preprocess_batch(astro_objects)
        self.feature_extractor.compute_features_batch(astro_objects, progress_bar=False)

        # Guardar resultados en CSVs por objeto usando función externa
        #batch_folder = save_astro_objects_to_csvs(astro_objects, messages_to_process, base_folder="csvs")
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
