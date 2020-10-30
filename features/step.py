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

from lc_classifier.features.custom import CustomStreamHierarchicalExtractor

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
        if not step_args.get("test_mode", False):
            self.insert_step_metadata()


    def insert_step_metadata(self):
        self.db.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_ID"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["STEP_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )

    def preprocess_xmatches(self, xmatches):
        """
        As of version 1.0.0 it does no preprocess operations on xmatches.

        Parameters
        ----------
        xmatches : dict
            xmatches as they come from preprocess step
        """
        return xmatches

    def preprocess_metadata(self, metadata):
        """
        As of version 1.0.0 it does no preprocess operations on alert metadata.

        Parameters
        ----------
        metadata : dict
            metadata as they come from preprocess step
        """
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
            if not created:
                self.db.query().update(feature, {"value": feature.value})
        self.db.session.commit()

    def get_fid(self, feature):
        """
        Gets the band number (fid) of a feature.
        Most features include the fid in the name as a sufix after '_' (underscore) character.
        Some features don't include the fid in their name but are known to be asociated with a specific band or multiband.
        This method considers all of these cases and the possible return values are:

        - 0: for wise features and sgscore
        - 12: for multiband features or power_rate
        - 1: for fid = 1
        - 2 for fid = 2
        - -99 if there is no known band for the feature

        Parameters
        ----------
        feature : str
            name of the feature
        """
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

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} messages.")

        self.logger.info("Getting batch alert data")
        alert_data = pd.DataFrame([
                {"oid": message.get("oid"), "candid": message.get("candid", np.nan)}
                for message in messages ])
        self.logger.info(f"Find {len(alert_data.oid.unique())} Objects.")


        self.logger.info("Getting detections and non_detections")
        detections = []
        non_detections = []
        metadata = []
        xmatches = []

        for message in messages:
            detections.extend(message.get("detections", []))
            non_detections.extend(message.get("non_detections", []))
            metadata.append({
                        "oid": message["oid"],
                        "candid": message["candid"],
                        "sgscore1": message["metadata"]["ps1"]["sgscore1"]
                        })

            if "xmatches" in message and message["xmatches"] is not None:
                allwise = message["xmatches"].get("allwise")
                xmatch_values = {
                    "W1mag": allwise["W1mag"],
                    "W2mag": allwise["W2mag"],
                    "W3mag": allwise["W3mag"]
                }

            else:
                xmatch_values = {
                    "W1mag": np.nan,
                    "W2mag": np.nan,
                    "W3mag": np.nan
                }
            xmatches.append({
                        "oid": message["oid"],
                        "candid": message["candid"],
                        **xmatch_values
            })


        metadata = pd.DataFrame(metadata)
        xmatches = pd.DataFrame(xmatches)
        detections = pd.DataFrame(detections)
        non_detections = pd.DataFrame(non_detections)
        detections.drop_duplicates(["oid", "candid"], inplace=True)
        non_detections.drop_duplicates(["oid", "mjd", "fid"], inplace=True)
        detections.set_index("oid", inplace=True)
        non_detections.set_index("oid", inplace=True)

        self.logger.info(f"Calculating features")
        features = self.compute_features(detections, non_detections, metadata, xmatches)

        self.logger.info(f"Features calculated: {features.shape}")

        # if self.producer:
        #     out_message = {"features": result, "candid": message["candid"], "oid": oid}
        #     self.producer.produce(out_message, key=oid)
