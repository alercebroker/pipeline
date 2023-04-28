import logging
import warnings
import numpy as np
import pandas as pd

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
        
        scribe_class = self.get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
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
        commands = parse_scribe_payload(features, self.feature_version.version) # version form metadata check
        for command in commands:
            self.scribe_producer.produce(command, key="algo")

    def produce_temp(self, features: pd.DataFrame, alert_data: pd.DataFrame):
        output = parse_output(features, alert_data)
        for output_message in output:
            self.producer.produce(output_message, key="algo")

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} messages.")

        preprocess_id = messages[0]["preprocess_step_id"]
        self.insert_feature_version(preprocess_id)

        self.logger.info("Getting batch alert data")
        
        detections, non_detections = [], []

        for message in messages:
            msg_detections = message.get("detections")
            msg_non_detections = message.get("non_detections")
            
            if msg_detections:
                detections.append(msg_detections)
            
            if msg_non_detections:
                non_detections.append(msg_non_detections)

        self.logger.info(f"Calculating features")
        features = self.compute_features(
            detections, non_detections, [], [], []
        )
        self.logger.info(f"Features calculated: {features.shape}")
        if len(features) > 0:
            self.produce_to_scribe(features)
        if self.producer:
            self.produce(features, messages)
