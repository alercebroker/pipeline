from apf.core.step import GenericStep
import logging
from apf.core import get_class
from apf.producers import KafkaProducer

from db_plugins.db.generic import new_DBConnection
from db_plugins.db.models import (
    Object,
    Detection,
)
from db_plugins.db.mongo.connection import MongoDatabaseCreator

import numpy as np
import pandas as pd

class GenericSaveStep(GenericStep):
    """GenericSaveStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """
    def __init__(self, consumer = None, config = None, level = logging.INFO, producer=None, db_connection=None,
                 **step_args):
        super().__init__(consumer,config=config, level=level)
        
        if "CLASS" in config["PRODUCER_CONFIG"]:
            Producer = get_class(config["PRODUCER_CONFIG"]["CLASS"])
        else:
            Producer = KafkaProducer

        self.producer = producer or Producer(config["PRODUCER_CONFIG"])
        self.driver = db_connection or new_DBConnection(MongoDatabaseCreator)
        self.driver.connect(config["DATABASE"])
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.logger.info(f"SAVE {self.version}")

    """
    def insert_step_metadata(self):
        self.driver.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_ID"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["STEP_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )
        self.driver.session.commit()
    """

    def get_objects(self, oids):
        query = self.driver.query(Object).filter(Object.oid.in_(oids))
        return pd.read_sql(query.statement, self.driver.engine)

    def remove_stamps(self, alerts):
        for k in ["cutoutDifference", "cutoutScience", "cutoutTemplate"]:
            del alerts[k]

    def generate_aid(self, ra, dec):
        pass

    def preprocess_detections(self, detections) -> None:
        detections.loc[:, "mjd"] = detections["jd"] - 2400000.5
        detections.loc[:, "has_stamp"] = True
        detections.loc[:, "step_id"] = self.version
        return detections
    
    def insert_objects(self, objects):
        new_objects = objects["new"]
        objects.drop_duplicates(["oid"], inplace=True)
        objects.drop(columns=["new"], inplace=True)

        to_insert = objects[new_objects]
        to_update = objects[~new_objects]

        if len(to_insert) > 0:
            self.logger.info(f"Inserting {len(to_insert)} new objects")
            to_insert.replace({np.nan: None}, inplace=True)
            dict_to_insert = to_insert.to_dict("records")
            self.driver.query().bulk_insert(dict_to_insert, Object)

        if len(to_update) > 0:
            self.logger.info(f"Updating {len(to_update)} objects")
            to_update.replace({np.nan: None}, inplace=True)
            to_update.rename(columns={"oid": "_oid"}, inplace=True)
            dict_to_update = to_update.to_dict("records")
            self.driver.query().update(dict_to_update, Object)
            
    def insert_detections(self, detections):
        self.logger.info(f"Inserting {len(detections)} new detections")
        detections = detections.where(detections.notnull(), None)
        dict_detections = detections.to_dict("records")
        self.driver.query().bulk_insert(dict_detections, Detection)

    def apply_objstats_from_correction(df):
        response = {}
        df_mjd = df.mjd
        idxmax = df_mjd.values.argmax()
        df_max = df.iloc[idxmax]
        df_ra = df.ra
        df_dec = df.dec
        response["meanra"] = df_ra.mean()
        response["meandec"] = df_dec.mean()
        response["sigmara"] = df_ra.std()
        response["sigmadec"] = df_dec.std()
        response["firstmjd"] = df_mjd.min()
        response["lastmjd"] = df_max.mjd

        return pd.Series(response)

    def preprocess_objects(self, objects, detections, alerts):
        oids = objects.oid.unique()
        apply_last_alert = lambda x: self.get_last_alert(x)
        last_alerts = alerts.groupby("oid", sort=False).apply(apply_last_alert)
        last_alerts.drop(columns=["oid"], inplace=True)

        detections_last_alert = detections.join(last_alerts, on="oid")
        detections_last_alert["objectId"] = detections_last_alert.oid
        detections_last_alert.drop_duplicates(["candid", "oid"], inplace=True)
        detections_last_alert.reset_index(inplace=True)

        new_objects = detections_last_alert.groupby("objectId").apply(self.apply_objstats_from_correction)
        new_objects.reset_index(inplace=True)

        new_names = dict(
            [(col, col.replace("-", "_")) for col in new_objects.columns if "-" in col]
        )

        new_objects.rename(columns={"objectId": "oid", **new_names}, inplace=True)
        new_objects["new"] = ~new_objects.oid.isin(oids)

        return new_objects

    def execute(self, messages):
        global MAGSTATS_UPDATE_KEYS
        global MAGSTATS_UPDATE_KEYS_STMT
        self.logger.info(f"Processing {len(messages)} alerts")

        # Casting to a dataframe
        self.logger.info(f"Preprocessing alerts")

        alerts = pd.DataFrame(messages)
        alerts.drop_duplicates("candid", inplace=True)
        alerts.reset_index(inplace=True)
        # Removing stamps
        self.remove_stamps(alerts)

        # Getting just the detections
        self.logger.info(f"Doing correction to detections")
        detections = pd.DataFrame(list(alerts["candidate"]))
        detections.loc[:, "survey_id"] = alerts["objectId"]
        detections.loc[:, "alerce_id"] = self.generate_aid(alerts["ra"], alerts["dec"])
        detections = self.preprocess_detections(detections)

        # Getting data from database, and processing prv_candidates
        self.logger.info(f"Processing light curves")
        light_curves = self.preprocess_lightcurves(detections, alerts)

        # Getting other tables
        objects = self.get_objects(detections["oid"].unique())
        objects = self.preprocess_objects(objects, light_curves, detections)

        # Insert new objects and update old objects
        self.insert_objects(objects)
        new_detections = light_curves["detections"]["new"]
        self.insert_detections(light_curves["detections"].loc[new_detections])

        self.produce(alerts, light_curves)

        del alerts
        del detections
        del light_curves