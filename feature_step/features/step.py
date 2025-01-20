import pandas as pd
import logging
import json
from typing import Any, Dict, Iterable, List, Optional

from apf.core import get_class
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer

from lc_classifier.features.core.base import AstroObject
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor

from .database import (
    PSQLConnection,
    get_sql_references,
)

from .utils.metrics import get_sid
from .utils.parsers import parse_output, parse_scribe_payload
from .utils.parsers import detections_to_astro_object

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
        self.lightcurve_preprocessor = ZTFLightcurvePreprocessor(drop_bogus=True)
        self.feature_extractor = ZTFFeatureExtractor()

        scribe_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = scribe_class(self.config["SCRIBE_PRODUCER_CONFIG"])
        self.extractor_version = version("feature-step")
        self.extractor_group = ZTFFeatureExtractor.__name__

        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.FeatureStep")

        self.min_detections_features = config.get("MIN_DETECTIONS_FEATURES", None)
        if self.min_detections_features is not None:
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

        count_fatures = 0
        flush = False
        for command in update_features_cmds:
            count_fatures += 1
            if count_fatures == len(update_features_cmds):
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
        if self.min_detections_features is None:
            return messages

        def has_enough_detections(message: dict) -> bool:
            n_dets = len([True for det in message["detections"] if not det["forced"]])
            return n_dets >= self.min_detections_features

        filtered_messages = filter(has_enough_detections, messages)
        filtered_messages = list(filtered_messages)
        return filtered_messages

    def execute(self, messages):
        candids = {}
        astro_objects = []
        messages_to_process = []

        oids = set()
        for msg in messages:
            oids.add(msg["oid"])
        references_db = self._get_sql_references(list(oids))

        for message in messages:
            if not message["oid"] in candids:
                candids[message["oid"]] = []
            candids[message["oid"]].extend(message["candid"])
            m = map(
                lambda x: {**x, "index_column": str(x["candid"]) + "_" + x["oid"]},
                message.get("detections", []),
            )

            xmatch_data = message["xmatches"]

            ao = detections_to_astro_object(list(m), xmatch_data, references_db)
            astro_objects.append(ao)
            messages_to_process.append(message)

        self.lightcurve_preprocessor.preprocess_batch(astro_objects)
        self.feature_extractor.compute_features_batch(astro_objects, progress_bar=False)

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
