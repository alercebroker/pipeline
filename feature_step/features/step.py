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
        # Bogus detections are dropped in pre_execute
       
        scribe_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = scribe_class(self.config["SCRIBE_PRODUCER_CONFIG"])

        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.FeatureStep")
        self.survey = self.config.get("SURVEY")
        
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
            self.extractor_version = get_or_create_version_id(
                self.db_sql, self.schema, version_name, self.logger
            )
            
            # Fetch feature name lookup table from multisurvey schema
            self.feature_name_lut = get_feature_name_lut(
                self.db_sql, self.schema, self.logger
            )

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

    def expand_messages_by_detection_count(self, messages: List[dict]) -> List[dict]:
        """
        Expande cada mensaje en múltiples mensajes con cantidades incrementales de detecciones.
        Por cada mensaje original con N detecciones, genera N mensajes:
        - Mensaje 1: 1 detección
        - Mensaje 2: 2 detecciones
        - ...
        - Mensaje N: N detecciones
        
        Args:
            messages: Lista de mensajes originales
            
        Returns:
            Lista expandida de mensajes con detecciones incrementales
        """
        expanded_messages = []
        
        for message in messages:
            detections = message.get("detections", [])
            num_detections = len(detections)
            
            # Generar un mensaje por cada cantidad de detecciones (1 a N)
            for i in range(1, num_detections + 1):
                expanded_message = message.copy()
                expanded_message["detections"] = detections[:i]
                expanded_messages.append(expanded_message)
        
        return expanded_messages

    def count_class_objects(self, base_folder: str = "csvs") -> dict:
        """
        Cuenta el número de objetos (carpetas) por cada clase.
        
        Args:
            base_folder: Carpeta base donde se encuentran los CSVs
            
        Returns:
            Diccionario con {class_name: num_objects}
        """
        import os
        class_counts = {}
        
        if not os.path.exists(base_folder):
            return class_counts
        
        for class_name in os.listdir(base_folder):
            class_path = os.path.join(base_folder, class_name)
            if os.path.isdir(class_path):
                # Contar subcarpetas (objetos)
                num_objects = len([d for d in os.listdir(class_path) 
                                  if os.path.isdir(os.path.join(class_path, d))])
                class_counts[class_name] = num_objects
        
        return class_counts

    def pre_execute(self, messages: List[dict]):

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
        
        
        filtered_messages = list(filter(has_enough_detections, filtered_messages))

        # Contar objetos por clase
        class_counts = self.count_class_objects(base_folder="csvs")
        
        # Reportar conteo actual detallado
        #self.logger.info("="*60)
        #self.logger.info("📊 REPORTE DE OBJETOS POR CLASE:")
        #for class_name in ['EB', 'Mira', 'RRL']:
        #    count = class_counts.get(class_name, 0)
        #    status = "✅ OK" if count < 500 else "🛑 LÍMITE ALCANZADO"
        #    self.logger.info(f"  {class_name:10s}: {count:4d}/500 objetos {status}")
        #self.logger.info("="*60)
        
        # Filtrar solo mensajes con class_name específicas Y que tengan menos de 500 objetos
        def has_valid_class_name(message: dict) -> bool:
            class_name = message.get('class_name')
            if class_name not in ['EB', 'Mira', 'RRL']:
                return False
                #pass
            # Verificar que la clase tenga menos de 500 objetos
            num_objects = class_counts.get(class_name, 0)
            if num_objects >= 3000:
                self.logger.warning(f"⚠️  Clase {class_name} tiene {num_objects} objetos, excede el límite de 500. Descartando mensaje.")
                return False
            return True
        
        filtered_messages = list(filter(has_valid_class_name, filtered_messages))

        # EXPANDIR MENSAJES: Comentar esta línea para desactivar la expansión
        #filtered_messages = self.expand_messages_by_detection_count(filtered_messages)

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
            db_references = get_sql_references(
                list(oids), self.db_sql, keys=["oid", "rfid", "sharpnr", "chinr"]
            )
            references_db = db_references[db_references["chinr"] >= 0.0].copy()
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
                ao = self.detections_to_astro_object_fn(list(m), xmatch_data,references_db)
            else:
                forced = message.get("forced_sources", None) #si no hay detections, filtrar forced photometry
                ao = self.detections_to_astro_object_fn(list(m), forced)

            ao.class_name = message.get("class_name", None)
            ao.len_detections = len(message.get("detections", []))
            astro_objects.append(ao)
            messages_to_process.append(message)

        self.lightcurve_preprocessor.preprocess_batch(astro_objects)
        self.feature_extractor.compute_features_batch(astro_objects, progress_bar=False)

        # Guardar resultados en CSVs por objeto usando función externa
        batch_folder = save_astro_objects_to_csvs(astro_objects, messages_to_process, base_folder="csvs")
        
        # Reportar estado actualizado después de guardar
        class_counts_after = self.count_class_objects(base_folder="csvs")
        self.logger.info("📝 Estado actualizado después de guardar:")
        for class_name in ['EB', 'Mira', 'RRL']:
            count = class_counts_after.get(class_name, 0)
            self.logger.info(f"  {class_name}: {count} objetos")
        
        #self.produce_to_scribe(astro_objects)
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
