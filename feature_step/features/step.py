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

    def produce_to_scribe(self, features: pd.DataFrame, oids={}):
        commands = parse_scribe_payload(features, self.features_extractor, oids=oids)

        for command in commands:
            self.scribe_producer.produce({"payload": json.dumps(command)})

    def execute(self, messages):
        detections, non_detections, xmatch = [], [], []
        oids = {}
        for message in messages:
            # need to create a hash to store aid to oids relations
            dets = message.get("detections", [])
            for det in dets:
                if det["sid"] != "ZTF":
                    continue
                _oids = oids.get(det["aid"], [])
                if det["oid"] not in _oids:
                    _oids.append(det["oid"])
                    oids[det["aid"]] = _oids    

            detections.extend(dets)

            # same thing for detections
            non_dets = message.get("non_detections", [])
            for nd in non_dets:
                if nd["sid"] != "ZTF":
                    continue
                
                _oids = oids.get(nd["aid"], [])
                if det["oid"] not in _oids:
                    _oids.append(nd["oid"])
                    oids[nd["aid"]] = _oids

            non_detections.extend(non_dets)
            xmatch.append(
                {"aid": message["aid"], **(message.get("xmatches", {}) or {})}
            )

        features_extractor = self.features_extractor(
            detections, non_detections, xmatch
        )
        features = features_extractor.generate_features()

        if len(features) > 0:
            self.produce_to_scribe(features, oids=oids)

        output = parse_output(features, messages, self.features_extractor)
        return output

    def post_execute(self, result):
        self.metrics["sid"] = get_sid(result)
        return result
