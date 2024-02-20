import json
from typing import Any, Callable, Dict, Iterable

import pandas as pd
from apf.core import get_class
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer

from features.core.elasticc import ELAsTiCCFeatureExtractor
from features.core.ztf import ZTFFeatureExtractor
from features.core.handlers.detections import NoDetectionsException
from features.utils.metrics import get_sid
from features.utils.parsers import parse_output, parse_scribe_payload


class FeaturesComputer(GenericStep):
    """FeaturesComputer Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(
        self,
        extractor: type[ZTFFeatureExtractor]
        | Callable[..., ELAsTiCCFeatureExtractor],
        config=None,
        **step_args,
    ):
        super().__init__(config=config, **step_args)
        self.features_extractor = extractor

        scribe_class = get_class(
            self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"]
        )
        self.scribe_producer = scribe_class(
            self.config["SCRIBE_PRODUCER_CONFIG"]
        )

    def produce_to_scribe(self, features: pd.DataFrame):
        commands = parse_scribe_payload(features, self.features_extractor)

        count = 0
        flush = False
        for command in commands:
            count += 1
            if count == len(commands):
                flush = True
            self.scribe_producer.produce(
                {"payload": json.dumps(command)}, flush=flush
            )

    def pre_produce(self, result: Iterable[Dict[str, Any]] | Dict[str, Any]):
        self.set_producer_key_field("oid")
        return result

    def execute(self, messages):
        detections, non_detections, xmatch = [], [], []
        candids = {}

        for message in messages:
            if not message["oid"] in candids:
                candids[message["oid"]] = []
            candids[message["oid"]].extend(message["candid"])
            detections.extend(message.get("detections", []))
            non_detections.extend(message.get("non_detections", []))
            xmatch.append(
                {"oid": message["oid"], **(message.get("xmatches", {}) or {})}
            )

        try:
            features_extractor = self.features_extractor(
                detections, non_detections, xmatch
            )
        except NoDetectionsException:
            return []

        features = features_extractor.generate_features()
        if len(features) > 0:
            self.produce_to_scribe(features)

        output = parse_output(
            features, messages, self.features_extractor, candids
        )
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
