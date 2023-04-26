import logging
import datetime
import warnings
import numpy as np
import pandas as pd

from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from lc_classifier.features.custom import CustomStreamHierarchicalExtractor
from sqlalchemy.sql.expression import bindparam
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

    def insert_feature_version(self, preprocess_id): # pendiente
        """
        self.feature_version, created = self.db.query(FeatureVersion).get_or_create(
            filter_by={
                "version": self.config["STEP_METADATA"]["FEATURE_VERSION"],
                "step_id_feature": self.config["STEP_METADATA"]["STEP_ID"],
                "step_id_preprocess": preprocess_id,
            }
        )
        """

    def get_xmatches_from_message(self, message):
        if "xmatches" in message and message["xmatches"] is not None:
            xmatch_values = {
                "W1mag": message["W1mag"],
                "W2mag": message["W2mag"],
                "W3mag": message["W3mag"],
            }
        else:
            xmatch_values = {"W1mag": np.nan, "W2mag": np.nan, "W3mag": np.nan}
        return {"aid": message["aid"], **xmatch_values}

    def delete_duplicate_detections(self, detections):
        self.logger.debug(f"Before Dropping: {len(detections)} Detections")
        detections.drop_duplicates(["oid", "candid"], inplace=True)
        self.logger.debug(f"After Dropping: {len(detections)} Detections")

    def delete_duplicate_non_detections(self, non_detections):
        self.logger.debug(f"Before Dropping: {len(non_detections)} Non Detections")
        non_detections["round_mjd"] = non_detections.mjd.round(6)
        non_detections.drop_duplicates(["oid", "round_mjd", "fid"], inplace=True)
        self.logger.debug(f"After Dropping: {len(non_detections)} Non Detections")

    def delete_duplicates(self, detections, non_detections):
        self.delete_duplicate_detections(detections)
        self.delete_duplicate_non_detections(non_detections)

    def get_data_from_messages(self, messages):
        """
        Gets detections, non_detections, metadata and xmatches from consumed messages
        and converts them to pd.DataFrame
        """
        detections = []
        non_detections = []
        metadata = []
        xmatches = []

        for message in messages:
            detections.extend(message.get("detections", []))
            non_detections.extend(message.get("non_detections", []))
            xmatches.append(self.get_xmatches_from_message(message))

        return (
            pd.DataFrame(detections),
            pd.DataFrame(non_detections),
            pd.DataFrame(metadata),
            pd.DataFrame(xmatches),
        )

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
        alert_data = pd.DataFrame(
            [
                {
                    "aid": message.get("aid"),
                    "meanra": message.get("meanra"),
                    "meandec": message.get("meandec"),
                    "detections": message.get("detections", []),
                    "non_detections": message.get("non_detections", []),
                    "xmatches": message.get("xmatches", [])
                }
                for message in messages
            ]
        )
        unique_aid = alert_data.aid.unique()
        qty_oid = len(unique_aid)

        self.logger.info(f"Found {qty_oid} Objects.")

        self.logger.info("Getting detections and non_detections")

        detections, non_detections, metadata, xmatches = self.get_data_from_messages(
            messages
        )

        if qty_oid < len(messages):
            self.delete_duplicates(detections, non_detections)

        if len(detections):
            detections.set_index("aid", inplace=True)
        if len(non_detections):
            non_detections.set_index("aid", inplace=True)

        self.logger.info(f"Calculating features")
        features = self.compute_features(
            detections, non_detections, metadata, xmatches, {"object": None}
            #objects se pedia de db. la info seguramente viene en el message
        )
        self.logger.info(f"Features calculated: {features.shape}")
        if len(features) > 0:
            self.produce_to_scribe(features)
        if self.producer:
            self.produce(features, alert_data)
