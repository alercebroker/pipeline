"""Feature step: messages -> AstroObjects -> features -> scribe commands and output messages.

Everything that depends on the survey (which epochs a message carries, how the
AstroObject is built, which scribe commands and output keys are produced) is
asked to the Survey object that features.surveys.registry returns for SURVEY.
"""
import pandas as pd  # type of the references table
import logging  # step logger
import json  # encode the scribe commands
from typing import Any, Dict, Iterable, List, Optional  # type hints of the methods

from apf.core import get_class  # resolve the scribe producer class named in the config
from apf.core.step import GenericStep  # base class of every step
from apf.consumers import KafkaConsumer  # consumer type checked in tear_down

from lc_classifier.features.core.base import AstroObject  # object the features are computed on

from xmatch_client import XmatchClient  # client of the xmatch service

from .database import (  # database access of the step
    PSQLConnection,  # connection type of db_sql
    get_sql_references,  # references of a list of oids
    get_feature_name_lut,  # feature id -> name table
    get_or_create_version_id,  # id of the running feature-step version
)  # end of the database imports

from .surveys.registry import get_survey  # SURVEY name -> Survey object

from importlib.metadata import version  # installed version of the feature-step package


def _match_for_output(match: Dict[str, Any]) -> Dict[str, Any]:
    """Shape a service match for the output topic: keep only the map-valued
    `metadata` entries.

    The xmatch service returns each metadata row with its own `id` column, a
    plain string equal to `match_id`. The output schema
    (`schemas/feature_ms_step/ztf/xmatch.avsc`) declares `metadata` as a map of
    maps, so any scalar entry makes Avro serialization fail. Nothing downstream
    reads the scalars: the parser only uses the `w*mpro` maps. The scribe path
    receives the untouched match list, not this copy.
    """
    metadata = match.get("metadata") or {}  # None or missing counts as empty
    return {  # copy of the match with the filtered metadata
        **match,  # every other key as received
        "metadata": {k: v for k, v in metadata.items() if isinstance(v, dict)},  # map-valued entries only
    }  # end of the shaped match


class FeatureStep(GenericStep):
    """Step that computes light curve features for the survey named in SURVEY.

    Parameters
    ----------
    config : dict
        Step configuration. SURVEY must be a name known to
        features.surveys.registry; a missing or unknown name raises ValueError.
    db_sql : PSQLConnection
        Database connection used for the references, the version id and the
        feature name lookup table.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(  # signature spans several lines
        self,  # step instance
        config=None,  # step configuration
        db_sql: PSQLConnection = None,  # database connection
        **step_args,  # passed through to GenericStep
    ):  # end of the signature
        """Build the scribe producer, the survey objects, the version id, the LUT and the xmatch client."""

        super().__init__(config=config, **step_args)  # consumer, producer and self.config

        scribe_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])  # producer class named in the config
        self.scribe_producer = scribe_class(self.config["SCRIBE_PRODUCER_CONFIG"])  # producer of the scribe commands

        self.scribe_topic_name = self.config["SCRIBE_PRODUCER_CONFIG"].get("TOPIC")  # topic of the scribe commands
        self.db_sql = db_sql  # connection for references, version id and LUT
        self.logger = logging.getLogger("alerce.FeatureStep")  # step logger
        self.survey = get_survey(self.config.get("SURVEY"))  # Survey object; ValueError when missing or unknown
        self.survey_name = self.survey.name  # the SURVEY string, sent in the xmatch scribe command
        self.use_xmatch = self.config.get("USE_XMATCH", False)  # whether to ask the xmatch service
        # Catalogs to cross-match against (one conesearch request per catalog).
        self.xmatch_catalogs = self.config.get(  # catalogs from the config
            "XMATCH_CATALOGS", ["allwise", "gaia"]  # key and default
        )  # end of the catalogs lookup

        # Get schema from configuration
        self.schema = self.config.get("DB_CONFIG", {}).get("SCHEMA", "multisurvey")  # db schema, multisurvey by default

        self.id_column = self.survey.id_column  # message key and epoch key holding the detection id
        self.lightcurve_preprocessor = self.survey.make_preprocessor()  # preprocessor of the survey
        self.feature_extractor = self.survey.make_extractor()  # feature extractor of the survey
        self.extractor_group = type(self.feature_extractor).__name__  # class name of the extractor

        # Get version name and resolve version_id from version_lut table
        version_name = version("feature-step")  # installed package version
        self.extractor_version = get_or_create_version_id(  # id of that version for this survey
            self.db_sql, self.schema, version_name, self.survey.sid, self.survey.tid, self.logger  # connection, schema, version, ids, logger
        )  # end of the version lookup

        # Fetch feature name lookup table from multisurvey schema
        self.feature_name_lut = get_feature_name_lut(  # feature id -> name for this survey
            self.db_sql, self.schema, self.survey.sid, self.survey.tid, self.logger  # connection, schema, ids, logger
        )  # end of the LUT read

        # Initialize xmatch client
        if self.use_xmatch:  # only when the xmatch is enabled
            xmatch_config = self.config.get("XMATCH_CONFIG", {})  # client arguments from the config
            self.xmatch_client = XmatchClient(**xmatch_config)  # client of the xmatch service

        self.min_detections_features = config.get("MIN_DETECTIONS_FEATURES", None)  # minimum non-forced epochs per object
        if self.min_detections_features is None:  # not configured
            self.min_detections_features = 1  # default: one detection
        else:  # configured
            self.min_detections_features = int(self.min_detections_features)  # may arrive as a string

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
        oids = []  # object ids to search, as strings
        ras = []  # right ascensions, same order
        decs = []  # declinations, same order

        for msg in messages:  # every incoming message
            if msg['sid'] == 2:  # messages of sid 2 are not cross-matched
                continue  # Saltar sid 2
            oids.append(str(msg["oid"]))  # oid as string
            ras.append(msg["meanra"])  # mean right ascension of the object
            decs.append(msg["meandec"])  # mean declination of the object

        # One request per catalog: the server scopes the search to each catalog
        # before the KNN, so we get the nearest match in every catalog instead
        # of only the single global nearest.
        results = []  # matches of every catalog
        for catalog in self.xmatch_catalogs:  # one request per catalog
            results += self.xmatch_client.conesearch_with_metadata(  # nearest match per object in this catalog
                ras=ras,  # right ascensions
                decs=decs,  # declinations
                oids=oids,  # object ids
                catalog=catalog,  # catalog to search
            )  # end of the request

        return results  # one match per (oid, catalog) found

    def produce_xmatch_to_scribe(self, xmatch_results: List[Dict[str, Any]], messages: List[dict],survey):
        """
        Produce xmatch results to scribe.

        Parameters
        ----------
        xmatch_results : List[Dict[str, Any]]
            List of xmatch results from conesearch_with_metadata
        messages : List[dict]
            Original messages to extract sid information
        survey : str
            Survey name written in each command
        """
        if not xmatch_results:  # nothing matched
            return  # nothing to send

        # Crear mapeo de oid -> sid desde los mensajes
        oid_to_sid = {}  # oid as string -> sid of its message
        for msg in messages:  # every incoming message
            oid = str(msg['oid'])  # oid as string, like the matches carry it
            sid = msg['sid']  # survey id of the message
            oid_to_sid[oid] = sid  # last message of an oid wins


        # Procesar cada resultado de xmatch
        flush = False  # set on the last match
        for idx, match in enumerate(xmatch_results):  # every match, with its position
            oid = str(match["oid"])  # oid of the match as string
            sid = oid_to_sid.get(oid, 1)  # Usar sid del mensaje o valor por defecto

            # Crear comando en el formato especificado
            xmatch_command = {  # scribe xmatch command
                'step': 'xmatch',  # scribe step
                'survey': survey,  # survey name
                'payload': {  # row to write
                    'oid': int(oid),  # object id
                    'sid': sid,  # survey id
                    'catalog': match.get('catalog', 'allwise'),  # catalog of the match
                    'dist': match.get('distance'),  # distance to the match
                    'oid_catalog': match.get('match_id')  # id of the match in its catalog
                }  # end of the payload
            }  # end of the command

            if idx == len(xmatch_results) - 1:  # last match of the batch
                flush = True  # flush after producing it

            self.scribe_producer.producer.produce(  # send the command
                topic= self.scribe_topic_name,  # scribe topic
                value=json.dumps(xmatch_command).encode("utf-8"),  # command as json bytes
                key=str(oid).encode("utf-8"),  # keyed by oid
            )  # end of the produce call

            if flush:  # after the last command
                self.scribe_producer.producer.flush()  # deliver the batch

    def produce_to_scribe(self, astro_objects: List[AstroObject]):
        """Send the scribe commands of the survey: the object updates first, then the feature upserts."""
        commands = self.survey.scribe_commands(  # commands built by the survey
            astro_objects,  # objects with computed features
            self.extractor_version,  # version id
            self.extractor_group,  # extractor class name
            self.feature_name_lut  # feature id -> name
        )  # end of the commands call

        update_object_cmds = commands.get("update_object", [])  # object updates, absent for some surveys
        update_features_cmds = commands["upserting_features"]  # feature upserts, always present

        count_objs = 0  # object updates sent so far
        flush = False  # set on the last object update
        for command in update_object_cmds:  # every object update
            count_objs += 1  # one more sent
            if count_objs == len(update_object_cmds):  # last object update
                flush = True  # flush after producing it

            oid = command["payload"]["oid"]  # object id of the command
            self.scribe_producer.producer.produce(  # send the command
                topic= self.scribe_topic_name,  # scribe topic
                value=json.dumps(command).encode("utf-8"),  # command as json bytes
                key=str(oid).encode("utf-8"),  # keyed by oid
            )  # end of the produce call

            if flush:  # after the last object update
                self.scribe_producer.producer.flush()  # deliver the batch

        count_features = 0  # feature upserts sent so far
        flush = False  # set on the last feature upsert
        for command in update_features_cmds:  # every feature upsert
            count_features += 1  # one more sent
            if count_features == len(update_features_cmds):  # last feature upsert
                flush = True  # flush after producing it

            oid = command["payload"]["oid"]  # object id of the command
            self.scribe_producer.producer.produce(  # send the command
                topic= self.scribe_topic_name,  # scribe topic
                value=json.dumps(command).encode("utf-8"),  # command as json bytes
                key=str(oid).encode("utf-8"),  # keyed by oid
            )  # end of the produce call

            if flush:  # after the last feature upsert
                self.scribe_producer.producer.flush()  # deliver the batch

    def pre_produce(self, result: Iterable[Dict[str, Any]] | Dict[str, Any]):
        """Key the output messages by oid and return them unchanged."""
        self.set_producer_key_field("oid")  # output topic key
        return result  # messages as execute returned them

    def _get_sql_references(self, oids: List[str]) -> Optional[pd.DataFrame]:
        """Return the references of `oids` read from the database, keeping chinr >= 0."""
        db_references = get_sql_references(  # references of the batch
            oids, self.db_sql, keys=["oid", "rfid", "sharpnr", "chinr"], schema=self.schema  # oids, connection, columns, schema
        )  # end of the query
        db_references = db_references[db_references["chinr"] >= 0.0].copy()  # drop references without a valid chinr
        return db_references  # references table

    def pre_execute(self, messages: List[dict]):
        """Attach the xmatch to each message, gather its epochs under "detections" and drop the objects with too few."""

        # Para LSST: obtener xmatch info para TODOS los mensajes antes de filtrar
        if self.use_xmatch and len(messages) > 0:  # xmatch enabled and something to match
            xmatch_results = self.get_xmatch_info(messages)  # matches of every catalog
            # xmatch_results holds one match per (oid, catalog). Features only
            # use the allwise match (W1-W4), so map oid -> allwise match for the
            # per-message attach; the full list still goes to scribe below.
            xmatch_dict = {  # oid as string -> allwise match shaped for the output
                str(match['oid']): _match_for_output(match)  # scalar metadata entries dropped
                for match in xmatch_results  # every match
                if match.get('catalog') == 'allwise'  # allwise only
            }  # end of the allwise matches

            for msg in messages:  # every incoming message
                oid_str = str(msg['oid'])  # oid as string, like the matches carry it
                if oid_str in xmatch_dict:  # the object has an allwise match
                    msg['xmatches'] = xmatch_dict[oid_str]  # attach it
                else:  # no allwise match
                    msg['xmatches'] = None  # explicit None
            self.produce_xmatch_to_scribe(xmatch_results, messages, self.survey_name)  # every match goes to scribe

        filtered_messages = []  # copies of the messages with their epochs gathered
        for message in messages:  # every incoming message
            filtered_message = message.copy()  # shallow copy, the input message keeps its own keys
            filtered_message["detections"] = self.survey.epochs_from_message(message)  # epochs the survey keeps
            filtered_messages.append(filtered_message)  # candidate for the count filter

        def has_enough_detections(message: dict) -> bool:
            """Return whether the message has at least min_detections_features non-forced epochs."""
            n_dets = len([True for det in message["detections"] if not det.get("forced", False)])  # non-forced epochs
            return n_dets >= self.min_detections_features  # enough to compute features

        filtered_messages = list(filter(has_enough_detections, filtered_messages))  # drop objects with too few epochs

        return filtered_messages  # messages execute will process

    def execute(self, messages):
        """Build one AstroObject per message, compute its features, send the scribe commands and return the output."""
        candids = {}  # oid -> detection ids of its messages
        astro_objects = []  # one AstroObject per message
        messages_to_process = []  # messages in the same order as astro_objects

        oids = set()  # distinct oids of the batch
        for msg in messages:  # every prepared message
            oids.add(msg["oid"])  # collect its oid

        references_db = self._get_sql_references(list(oids)) if self.survey.needs_references else None  # db references, only for the surveys that use them

        for message in messages:  # every prepared message
            if not message["oid"] in candids:  # first message of this oid
                candids[message["oid"]] = []  # start its list
            candids[message["oid"]].extend(message[self.id_column]) #guarda los candid de cada oid
            m = map(  # epochs with their index_column, built lazily
                lambda x: {**x, "index_column": str(x[self.id_column]) + "_" + str(x["oid"])},  # "<detection id>_<oid>"
                message.get("detections", []),  # epochs gathered by pre_execute
            )  # end of the map

            ao = self.survey.build_astro_object(  # the survey builds the AstroObject
                list(m),  # epochs with index_column
                self.survey.forced_from_message(message),  # forced photometry the survey passes apart
                message.get("xmatches", None),  # allwise match or None
                references_db,  # db references or None
            )  # end of the build call
            astro_objects.append(ao)  # keep the object
            messages_to_process.append(message)  # and its message, same position

        self.lightcurve_preprocessor.preprocess_batch(astro_objects)  # clean the light curves in place
        self.feature_extractor.compute_features_batch(astro_objects, progress_bar=False)  # features written on each object
        self.produce_to_scribe(astro_objects)  # scribe commands
        output = self.survey.output_messages(astro_objects, messages_to_process, candids)  # one output message per object
        return output  # messages for the output topic

    def post_execute(self, result):
        """Remove the `reference` key from every output message and return the messages."""

        for message in result:  # every output message
            if "reference" in message:  # only some surveys carry it
                del message["reference"]  # not part of the output schema

        return result  # messages without reference

    def tear_down(self):
        """Close the consumer and the producer."""
        if isinstance(self.consumer, KafkaConsumer):  # a real Kafka consumer
            self.consumer.teardown()  # close it cleanly
        else:  # any other consumer
            self.consumer.__del__()  # release it
        self.producer.__del__()  # release the producer
