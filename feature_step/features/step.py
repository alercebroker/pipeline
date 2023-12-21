import json
from typing import Any, Callable, Dict, Iterable

import pandas as pd
from apf.core import get_class
from apf.core.step import GenericStep

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

    def produce_to_scribe(
        self, messages_aid_oid: dict, features: pd.DataFrame
    ):
        commands = parse_scribe_payload(
            messages_aid_oid, features, self.features_extractor
        )

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
        self.set_producer_key_field("aid")
        return result

    def execute(self, messages):
        detections, non_detections, xmatch, messages_aid_oid = [], [], [], {}
        candids = {}

        for message in messages:
            if not message["aid"] in candids:
                candids[message["aid"]] = []
            candids[message["aid"]].extend(message["candid"])
            detections.extend(message.get("detections", []))
            non_detections.extend(message.get("non_detections", []))
            xmatch.append(
                {"aid": message["aid"], **(message.get("xmatches", {}) or {})}
            )
            oids_of_aid = []
            oids_of_aid = [
                message_detection["oid"]
                for message_detection in message["detections"]
            ]
            messages_aid_oid[message["aid"]] = list(set(oids_of_aid))

        try:
            features_extractor = self.features_extractor(
                detections, non_detections, xmatch
            )
        except NoDetectionsException:
            return []

        features = features_extractor.generate_features()
        if len(features) > 0:
            self.produce_to_scribe(messages_aid_oid, features)

        output = parse_output(
            features, messages, self.features_extractor, candids
        )
        return output

    def post_execute(self, result):
        self.metrics["sid"] = get_sid(result)
        return result
