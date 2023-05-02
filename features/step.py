import logging
import warnings
import numpy as np
import pandas as pd

from apf.core import get_class
from apf.core.step import GenericStep
from lc_classifier.features.custom import CustomStreamHierarchicalExtractor
from features.utils.parsers import parse_scribe_payload, parse_output

warnings.filterwarnings("ignore")
logging.getLogger("GP").setLevel(logging.WARNING)


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
        config=None,
        preprocessor=None,
        features_computer=None,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)
        self.preprocessor = preprocessor  # Not used
        self.features_computer = (
            features_computer or CustomStreamHierarchicalExtractor()
        )
        
        scribe_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = scribe_class(self.config["SCRIBE_PRODUCER_CONFIG"])

    def compute_features(self, detections, non_detections, metadata, xmatches, objects):
        """Compute Hierarchical-Features in detections and non detections to `dict`.

        **Example:**

        Parameters
        ----------
        detections : pandas.DataFrame
            Detections of an object
        non_detections : pandas.DataFrame
            Non detections of an object
        metadata : pandas.DataFrame
            Metadata from the alert with other catalogs info
        xmatches : pandas.DataFrame
            Xmatches data from xmatch step
        objects : pandas.DataFrame
            Data information of each object
        """
        features = self.features_computer.compute_features(
            detections,
            non_detections=non_detections,
            metadata=metadata,
            xmatches=xmatches,
            objects=objects,
        )
        features = features.astype(float)
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        return features

    def produce_to_scribe(self, features: pd.DataFrame):
        commands = parse_scribe_payload(features, self.config["FEATURE_VERSION"]) # version form metadata check
        for command in commands:
            command_aid = command["criteria"]["_id"]
            self.scribe_producer.produce(command, key=command_aid)

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} messages.")

        self.logger.info("Getting batch alert data detections, non_detections and xmatches")
        detections, non_detections, xmatch = [], [], []

        for message in messages:
            #cambiar los detections y no detections
            msg_detections = message.get("detections")
            msg_non_detections = message.get("non_detections")
            detections.extend(msg_detections)
            non_detections.extend(msg_non_detections)
            xmatch.append(
                {
                    "aid": message["aid"],
                    **message["xmatches"]
                }
            )

        self.logger.info(f"Calculating features")
        features = self.features_computer.compute_features(
            detections, non_detections, xmatch, [], []
        )
        self.logger.info(f"Features calculated: {features.shape}")
        
        if len(features) > 0:
            self.produce_to_scribe(features)
   
        output = parse_output(features, messages)
        return output