import json

import pandas as pd

from apf.core import get_class
from apf.core.step import GenericStep

from features.utils.parsers import parse_scribe_payload, parse_output
from features.utils.metrics import get_sid

from features.core.ztf import ZTFFeatureExtractor
from features.core.elasticc import ELAsTiCCFeatureExtractor

from typing import Callable


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

        for command in commands:
            self.scribe_producer.produce({"payload": json.dumps(command)})

    def execute(self, messages):
        detections, non_detections, xmatch, messages_aid_oid = [], [], [], {}

        for message in messages:
            detections.extend(message.get("detections", []))
            non_detections.extend(message.get("non_detections", []))
            xmatch.append(
                {"aid": message["aid"], **(message.get("xmatches", {}) or {})}
            )
            oids_of_aid = []
            oids_of_aid = [
                message_detection["oid"]  for message_detection in message["detections"]
            ]
            messages_aid_oid[message["aid"]] = list(set(oids_of_aid))

        features_extractor = self.features_extractor(
            detections, non_detections, xmatch
        )
        features = features_extractor.generate_features()

        if len(features) > 0:
            self.produce_to_scribe(messages_aid_oid, features)

        output = parse_output(features, messages, self.features_extractor)
        return output

    def post_execute(self, result):
        self.metrics["sid"] = get_sid(result)
        return result
