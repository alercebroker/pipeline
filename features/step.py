import pandas as pd

from apf.core import get_class
from apf.core.step import GenericStep
from features.core.ztf import ZTFFeatureExtractor
from features.utils.parsers import parse_scribe_payload, parse_output


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
        extractor,
        config=None,
        **step_args,
    ):
        super().__init__(config=config, **step_args)
        self.features_extractor = extractor

        scribe_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = scribe_class(self.config["SCRIBE_PRODUCER_CONFIG"])

    def produce_to_scribe(self, features: pd.DataFrame):
        commands = parse_scribe_payload(
            features, self.features_extractor
        )

        for command in commands:
            command_aid = command["criteria"]["_id"]
            self.scribe_producer.produce(command, key=command_aid)

    def execute(self, messages):
        detections, non_detections, xmatch = [], [], []

        for message in messages:
            detections.extend(message.get("detections", []))
            non_detections.extend(message.get("non_detections", []))
            xmatch.append({"aid": message["aid"], **(message.get("xmatches", {}) or {})})

        features_extractor = self.features_extractor(detections, non_detections, xmatch)
        features = features_extractor.generate_features()

        if len(features) > 0:
            self.produce_to_scribe(features)

        output = parse_output(features, messages, self.features_extractor)
        return output
