import time
import logging
import datetime
import warnings
import numpy as np
import pandas as pd

from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from db_plugins.db.sql import SQLConnection
from db_plugins.db.sql.models import FeatureVersion, Object, Feature, Step

from late_classifier.features.custom import CustomStreamHierarchicalExtractor

from pandas.io.json import json_normalize

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
        consumer=None,
        config=None,
        preprocessor=None,
        features_computer=None,
        db_connection=None,
        producer=None,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(consumer, config=config, level=level)
        self.preprocessor = preprocessor  # Not used
        self.features_computer = (
            features_computer or CustomStreamHierarchicalExtractor()
        )
        self.db = db_connection or SQLConnection()
        self.db.connect(self.config["DB_CONFIG"]["SQL"])
        prod_config = self.config.get("PRODUCER_CONFIG", None)
        if prod_config:
            self.producer = producer or KafkaProducer(prod_config)
        else:
            self.producer = None
        self.db.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_VERSION"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["FEATURE_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )

    def preprocess_detections(self, detections):
        """Format a section of Kafka message that correspond of detections to `pandas.DataFrame`.
        This method take the key *detections* and use `json_normalize` to transform it to a DataFrame. After that
        rename some columns, put object_id like index and preprocess this detections.
        **Example:**

        Parameters
        ----------
        message : dict
            Message deserialized of Kafka.
        oid : string
            Object identifier of all detections
        """
        detections = self.create_detections_dataframe(detections)
        return detections

    def create_detections_dataframe(self, detections):
        detections = json_normalize(detections)
        detections.rename(
            columns={
                "alert.sgscore1": "sgscore1",
                "alert.isdiffpos": "isdiffpos",
            },
            inplace=True,
        )
        detections.set_index("oid", inplace=True)
        return detections

    def preprocess_non_detections(self, non_detections):
        return json_normalize(non_detections)

    def preprocess_xmatches(self, xmatches):
        return xmatches

    def preprocess_metadata(self, metadata):
        return metadata

    def compute_features(self, detections, non_detections, metadata, xmatches):
        """Compute Hierarchical-Features in detections and non detections to `dict`.

        **Example:**

        Parameters
        ----------
        detections : pandas.DataFrame
            Detections of an object
        non_detections : pandas.DataFrame
            Non detections of an object
        metadata : dict
            Metadata from the alert with other catalogs info
        obj : dict
            Object data
        """
        features = self.features_computer.compute_features(
            detections,
            non_detections=non_detections,
            metadata=metadata,
            xmatches=xmatches,
        )
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features = features.astype(float)
        return features

    def insert_db(self, oid, result, preprocess_id):
        """Insert the `dict` result in database.
        Consider:
            - object: Refer with oid
            - features: In result `dict`
            - version: Set in config of the step

        **Example:**

        Parameters
        ----------
        oid : string
            Object identifier of all detections
        result : dict
            Result of features compute
        fid : pd.DataFrame

        """
        feature_version, created = self.db.query(FeatureVersion).get_or_create(
            filter_by={
                "version": self.config["STEP_METADATA"]["FEATURE_VERSION"],
                "step_id_feature": self.config["STEP_METADATA"]["STEP_ID"],
                "step_id_preprocess": preprocess_id,
            }
        )
        if created:
            self.db.session.add(feature_version)
        for key in result:
            fid = self.get_fid(key)
            if fid < 0:
                continue
            feature, created = self.db.query(Feature).get_or_create(
                filter_by={
                    "oid": oid,
                    "name": key,
                    "fid": fid,
                    "version": feature_version.version,
                },
                value=result[key],
            )
            if created:
                self.db.session.add(feature)
            else:
                self.db.query().update(feature, {"value": feature.value})
        self.db.session.commit()

    def get_fid(self, feature):
        fid0 = [
            "W1",
            "W1-W2",
            "W2",
            "W2-W3",
            "W3",
            "W4",
            "g-W2",
            "g-W3",
            "g-r_ml",
            "gal_b",
            "gal_l",
            "r-W2",
            "r-W3",
            "rb",
            "sgscore1",
        ]
        fid12 = [
            "Multiband_period",
            "Period_fit",
            "g-r_max",
            "g-r_max_corr",
            "g-r_mean",
            "g-r_mean_corr",
        ]
        if feature in fid0:
            return 0
        if feature in fid12 or feature.startswith("Power_rate"):
            return 12
        fid = feature.rsplit("_", 1)[-1]
        if fid.isdigit():
            return int(fid)
        return -99

    def convert_nan(self, result):
        """Changes nan values to None

        Parameters
        ----------
        result : dict
            Dict that will have nans removed
        """
        cleaned_results = {}
        for key in result:
            if type(result[key]) is dict:
                cleaned_results[key] = self.convert_nan(result[key])
            else:
                if np.isnan(result[key]):
                    cleaned_results[key] = None
                else:
                    cleaned_results[key] = result[key]
        return cleaned_results

    def execute(self, message):
        oid = message["oid"]
        detections = self.preprocess_detections(message["detections"])
        non_detections = self.preprocess_non_detections(message["non_detections"])
        xmatches = self.preprocess_xmatches(message["xmatches"])
        metadata = self.preprocess_metadata(message["metadata"])
        if len(detections) < 6:
            self.logger.debug(f"{oid} Object has less than 6 detections")
            return
        self.logger.debug(f"{oid} Object has enough detections. Calculating Features")
        features = self.compute_features(detections, non_detections, metadata, xmatches)
        if len(features) <= 0:
            self.logger.debug(f"No features for {oid}")
            return
        if type(features) is pd.Series:
            features = pd.DataFrame([features])
        result = self.convert_nan(features.loc[oid].to_dict())
        self.insert_db(oid, result, message["preprocess_step_id"])
        if self.producer:
            out_message = {"features": result, "candid": message["candid"], "oid": oid}
            self.producer.produce(out_message, key=oid)
