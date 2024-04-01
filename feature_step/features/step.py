import json
from typing import Any, Callable, Dict, Iterable, Union, NewType, List

import pandas as pd
from apf.core import get_class
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer

from features.core.elasticc import ELAsTiCCFeatureExtractor
from features.core.ztf import ZTFFeatureExtractor
from features.utils.metrics import get_sid
from features.utils.parsers import parse_output, parse_scribe_payload


Candid = NewType("Candid", Union[str, int])


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
        extractor: (
            type[ZTFFeatureExtractor] | Callable[..., ELAsTiCCFeatureExtractor]
        ),
        config=None,
        **step_args,
    ):
        super().__init__(config=config, **step_args)
        self.features_extractor_factory = extractor

        scribe_class = get_class(
            self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"]
        )
        self.scribe_producer = scribe_class(
            self.config["SCRIBE_PRODUCER_CONFIG"]
        )

    def produce_to_scribe(self, features: pd.DataFrame):
        commands = parse_scribe_payload(
            features, self.features_extractor_factory
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
        self.set_producer_key_field("oid")
        for obj in result:
            for det in obj["detections"]:
                det.pop("rb", None)
        return result

    def preprare_candids(
        self, message: dict, candids: Dict[str, List[Candid]]
    ):
        """Prepares a dict where each key is the oid and value is a list of candid.

        Note: This method modifies the candids dictionary in place.

        Parameters
        ----------
        message : dict
            A message from the input stream.
        candids : dict
            A dictionary where each key is the oid and value is a list of candid.
        """
        if not message["oid"] in candids:
            candids[message["oid"]] = []
        candids[message["oid"]].extend(message["candid"])

    def prepare_detections(self, message: dict, detections: List[dict]):
        """Gets detections from a message and adds them to a list.

        Note: This method modifies the detections list in place.

        Parameters
        ----------
        message : dict
            A message from the input stream.
        detections : list
            A list of detections.
        """
        m = map(
            lambda x: {
                **x,
                "index_column": str(x["candid"]) + "_" + x["oid"],
                "rb": x["extra_fields"].get("rb", 0.0)
            },
            message.get("detections", []),
        )
        detections.extend(m)

    def prepare_non_detections(
        self, message: dict, non_detections: List[dict]
    ):
        """Gets non-detections from a message and adds them to a list.

        Note: This method modifies the non_detections list in place.

        Parameters
        ----------
        message : dict
            A message from the input stream.
        non_detections : list
            A list of non-detections.
        """
        non_detections.extend(message.get("non_detections", []))

    def prepare_xmatch(self, message: dict, xmatch: List[dict]):
        """Gets xmatch from a message and adds them to a list.

        Note: This method modifies the xmatch list in place.

        Parameters
        ----------
        message : dict
            A message from the input stream.
        xmatch : list
            A list of xmatch.
        """
        xmatch.append(
            {"oid": message["oid"], **(message.get("xmatches", {}) or {})}
        )

    def prepare_input(self, messages: Iterable[Dict[str, Any]]) -> tuple:
        """Prepares the input for the features extractor.

        Parameters
        ----------
        messages : Iterable[Dict[str, Any]]
            A list of messages from the input stream.

        Returns
        -------
        tuple
            A tuple containing the detections, non-detections, xmatch and candids.
        """
        detections, non_detections, xmatch = [], [], []
        candids = {}
        for message in messages:
            self.preprare_candids(message, candids)
            self.prepare_detections(message, detections)
            self.prepare_non_detections(message, non_detections)
            self.prepare_xmatch(message, xmatch)
        return detections, non_detections, xmatch, candids

    def execute(self, messages):
        detections, non_detections, xmatch, candids = self.prepare_input(
            messages
        )
        features_extractor = self.features_extractor_factory(
            detections, non_detections, xmatch
        )
        features = features_extractor.generate_features()
        if len(features) > 0:
            self.produce_to_scribe(features)
        output = parse_output(
            features, messages, self.features_extractor_factory, candids
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
