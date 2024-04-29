import logging
import json
from typing import Any, Callable, Dict, Iterable, List

from apf.core import get_class
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer

from lc_classifier.features.core.base import AstroObject
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor

from .utils.metrics import get_sid
from .utils.parsers import parse_output, parse_scribe_payload
from .utils.parsers import detections_to_astro_objects

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
        **step_args,
    ):

        super().__init__(config=config, **step_args)
        self.lightcurve_preprocessor = ZTFLightcurvePreprocessor()
        self.feature_extractor = ZTFFeatureExtractor()

        scribe_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = scribe_class(self.config["SCRIBE_PRODUCER_CONFIG"])
        self.extractor_version = version("feature-step")
        self.extractor_group = ZTFFeatureExtractor.__name__

    def produce_to_scribe(self, astro_objects: List[AstroObject]):
        commands = parse_scribe_payload(
            astro_objects,
            self.extractor_version,
            self.extractor_group,
        )

        count = 0
        flush = False
        for command in commands:
            count += 1
            if count == len(commands):
                flush = True
            self.scribe_producer.produce({"payload": json.dumps(command)}, flush=flush)

    def pre_produce(self, result: Iterable[Dict[str, Any]] | Dict[str, Any]):
        self.set_producer_key_field("oid")
        return result

    def execute(self, messages):
        candids = {}
        astro_objects = []
        messages_to_process = []
        for message in messages:
            if not message["oid"] in candids:
                candids[message["oid"]] = []
            candids[message["oid"]].extend(message["candid"])
            m = map(
                lambda x: {**x, "index_column": str(x["candid"]) + "_" + x["oid"]},
                message.get("detections", []),
            )

            xmatch_data = message["xmatches"]

            ao = detections_to_astro_objects(list(m), xmatch_data)
            astro_objects.append(ao)
            messages_to_process.append(message)

        self.lightcurve_preprocessor.preprocess_batch(astro_objects)
        self.feature_extractor.compute_features_batch(astro_objects, progress_bar=False)

        self.produce_to_scribe(astro_objects)
        output = parse_output(astro_objects, messages_to_process, candids)
        return output

    def post_execute(self, result):
        self.metrics["sid"] = get_sid(result)
        return result

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
