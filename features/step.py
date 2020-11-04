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
from sqlalchemy.sql.expression import bindparam

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

    def compute_features(self, detections, non_detections, metadata, xmatches):
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
        if not isinstance(feature, str):
            self.logger.error(
                f"Feature {feature} is not a valid feature. Should be str instance with fid after underscore (_)"
            )
            return -99
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
            "PPE",
        ]
        if feature in fid0:
            return 0
        if feature in fid12 or feature.startswith("Power_rate"):
            return 12
        fid = feature.rsplit("_", 1)[-1]
        if fid.isdigit():
            return int(fid)
        return -99

    def get_on_db(self, result):
        oids = result.index.values
        query = (
            self.db.query(Feature.oid)
            .filter(Feature.oid.in_(oids))
            .filter(Feature.version == self.config["STEP_METADATA"]["STEP_VERSION"])
            .distinct()
        )
        return pd.read_sql(query.statement, self.db.engine).oid.values

    def insert_feature_version(self, preprocess_id):
        self.feature_version, created = self.db.query(FeatureVersion).get_or_create(
            filter_by={
                "version": self.config["STEP_METADATA"]["FEATURE_VERSION"],
                "step_id_feature": self.config["STEP_METADATA"]["STEP_ID"],
                "step_id_preprocess": preprocess_id,
            }
        )

    def update_db(self, to_update, out_columns, apply_get_fid):
        if len(to_update) == 0:
            return
        self.logger.info(f"Updating {len(to_update)} features")
        to_update.replace({np.nan: None}, inplace=True)
        to_update = to_update.stack(dropna=False)
        to_update = to_update.to_frame()
        to_update.reset_index(inplace=True)
        to_update.columns = out_columns
        to_update["fid"] = to_update["name"].apply(apply_get_fid)
        to_update["version"] = self.feature_version.version
        to_update["name"] = to_update["name"].apply(lambda x: self.check_feature_name(x))
        to_update.rename(
            columns={
                "oid": "_oid",
                "fid": "_fid",
                "version": "_version",
                "name": "_name",
                "value": "_value",
            },
            inplace=True,
        )
        dict_to_update = to_update.to_dict("records")
        stmt = (
            Feature.__table__.update()
            .where(Feature.oid == bindparam("_oid"))
            .where(Feature.name == bindparam("_name"))
            .where(Feature.fid == bindparam("_fid"))
            .where(Feature.version == bindparam("_version"))
            .values(value=bindparam("_value"))
        )
        self.db.engine.execute(stmt, dict_to_update)
        return dict_to_update

    def insert_db(self, to_insert, out_columns, apply_get_fid):
        if len(to_insert) == 0:
            return
        self.logger.info(f"Inserting {len(to_insert)} new features")
        to_insert.replace({np.nan: None}, inplace=True)
        to_insert = to_insert.stack(dropna=False)
        to_insert = to_insert.to_frame()
        to_insert.reset_index(inplace=True)
        to_insert.columns = out_columns
        to_insert["fid"] = to_insert.name.apply(apply_get_fid)
        to_insert["version"] = self.feature_version.version
        to_insert["name"] = to_insert.name.apply(lambda x: self.check_feature_name(x))
        dict_to_insert = to_insert.to_dict("records")
        self.db.query().bulk_insert(dict_to_insert, Feature)
        return dict_to_insert

    def check_feature_name(self, name):
        fid = name.rsplit("_", 1)[-1]
        if fid.isdigit():
            return name.rsplit("_", 1)[0]
        else:
            return name

    def add_to_db(self, result):
        self.logger.info(result)
        """Insert the `dict` result in database.
        Consider:
            - object: Refer with oid
            - features: In result `dict`
            - version: Set in config of the step

        **Example:**

        Parameters
        ----------
        result : dict
            Result of features compute
        """
        out_columns = ["oid", "name", "value"]
        on_db = self.get_on_db(result)
        already_on_db = result.index.isin(on_db)
        to_insert = result.loc[~already_on_db]
        to_update = result.loc[already_on_db]
        apply_get_fid = lambda x: self.get_fid(x)

        if len(to_update) > 0:
            self.update_db(to_update, out_columns, apply_get_fid)
        if len(to_insert) > 0:
            self.insert_db(to_insert, out_columns, apply_get_fid)

    def produce(self, features, alert_data):
        alert_data.set_index("oid", inplace=True)
        alert_data.drop_duplicates(inplace=True, keep="last")
        features = features.join(alert_data)
        for oid, features_oid in features.iterrows():
            features_oid.replace({np.nan: None}, inplace=True)
            candid = features_oid.candid
            features_oid.drop(labels=["candid"], inplace=True)
            features_dict = features_oid.to_dict()
            out_message = {"features": features_dict, "oid": oid, "candid": candid}
            self.producer.produce(out_message, key=oid)

    def get_metadata_from_message(self, message):
        return {
            "oid": message["oid"],
            "candid": message["candid"],
            "sgscore1": message["metadata"]["ps1"]["sgscore1"],
        }

    def get_xmatches_from_message(self, message):
        if "xmatches" in message and message["xmatches"] is not None:
            allwise = message["xmatches"].get("allwise")
            xmatch_values = {
                "W1mag": allwise["W1mag"],
                "W2mag": allwise["W2mag"],
                "W3mag": allwise["W3mag"],
            }
        else:
            xmatch_values = {"W1mag": np.nan, "W2mag": np.nan, "W3mag": np.nan}
        return {"oid": message["oid"], "candid": message["candid"], **xmatch_values}

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
            metadata.append(self.get_metadata_from_message(message))
            xmatches.append(self.get_xmatches_from_message(message))

        return (
            pd.DataFrame(detections),
            pd.DataFrame(non_detections),
            pd.DataFrame(metadata),
            pd.DataFrame(xmatches),
        )

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} messages.")

        preprocess_id = messages[0]["preprocess_step_id"]
        self.insert_feature_version(preprocess_id)

        self.logger.info("Getting batch alert data")
        alert_data = pd.DataFrame(
            [
                {"oid": message.get("oid"), "candid": message.get("candid", np.nan)}
                for message in messages
            ]
        )
        unique_oid = len(alert_data.oid.unique())
        self.logger.info(f"Found {unique_oid} Objects.")

        self.logger.info("Getting detections and non_detections")

        detections, non_detections, metadata, xmatches = self.get_data_from_messages(
            messages
        )

        if unique_oid < len(messages):
            self.delete_duplicates(detections, non_detections)

        if len(detections):
            detections.set_index("oid", inplace=True)
        if len(non_detections):
            non_detections.set_index("oid", inplace=True)

        self.logger.info(f"Calculating features")
        features = self.compute_features(detections, non_detections, metadata, xmatches)

        self.logger.info(f"Features calculated: {features.shape}")
        if len(features) > 0:
            self.add_to_db(features)
        if self.producer:
            self.produce(features, alert_data)
