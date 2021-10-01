from apf.core.step import GenericStep
import logging
from apf.core import get_class
from apf.producers import KafkaProducer
import sys
sys.path.insert(0, '../../../../')

from db_plugins.db.generic import new_DBConnection
from db_plugins.db.models import (
    Object,
    Detection,
)
from db_plugins.db.mongo.connection import MongoDatabaseCreator

import numpy as np
import pandas as pd

class GeneratorID:
    @classmethod
    def generate_aid(self, ra, dec):
        # 19 Digit ID - two spare at the end for up to 100 duplicates
        id = 1000000000000000000

        # 2013-11-15 KWS Altered code to fix the negative RA problem
        if ra < 0.0:
            ra += 360.0

        if ra > 360.0:
            ra -= 360.0

        # Calculation assumes Decimal Degrees:

        ra_hh = int(ra / 15)
        ra_mm = int((ra / 15 - ra_hh) * 60)
        ra_ss = int(((ra / 15 - ra_hh) * 60 - ra_mm) * 60)
        ra_ff = int((((ra / 15 - ra_hh) * 60 - ra_mm) * 60 - ra_ss) * 100)

        if dec >= 0:
            h = 1
        else:
            h = 0
            dec = dec * -1

        dec_deg = int(dec)
        dec_mm = int((dec - dec_deg) * 60)
        dec_ss = int(((dec - dec_deg) * 60 - dec_mm) * 60)
        dec_f = int(((((dec - dec_deg) * 60 - dec_mm) * 60) - dec_ss) * 10)

        id += (ra_hh * 10000000000000000)
        id += (ra_mm * 100000000000000)
        id += (ra_ss * 1000000000000)
        id += (ra_ff * 10000000000)

        id += (h * 1000000000)
        id += (dec_deg * 10000000)
        id += (dec_mm * 100000)
        id += (dec_ss * 1000)
        id += (dec_f * 100)

        return id

class ParserSelector:
    parsers=[]

    def registerParser(self, parser):
        self.parsers.append(parser)

    def parse(self, alerts):
        for parser in self.parsers:
            if parser.canParseFile(alerts):
                return parser.parse(alerts)

import abc
class ParserGeneric(abc.ABC):
    @abc.abstractmethod
    def parse(self):
        """
        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass

    @abc.abstractmethod
    def canParseFile(self):
        """
        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass

class ParserATLAS(ParserGeneric):
    source = "ATLAS"

    @classmethod
    def parse(self, messages):
        for eachdict in messages:
            eachdict['candidate']['ra'] = eachdict['candidate']['RA']
            del eachdict['candidate']['RA']
            eachdict['candidate']['dec'] = eachdict['candidate']['Dec']
            del eachdict['candidate']['Dec']
            eachdict['candidate']['xpos'] = eachdict['candidate']['X']
            del eachdict['candidate']['X']
            eachdict['candidate']['ypos'] = eachdict['candidate']['Y']
            del eachdict['candidate']['Y']
            eachdict['candidate']['magpsf'] = eachdict['candidate']['Mag']
            del eachdict['candidate']['Mag']
            eachdict['candidate']['sigmapsf'] = eachdict['candidate']['Dmag']
            del eachdict['candidate']['Dmag']
            eachdict['candidate']['aimage'] = eachdict['candidate']['Major']
            del eachdict['candidate']['Major']
            eachdict['candidate']['bimage'] = eachdict['candidate']['Minor']
            del eachdict['candidate']['Minor']
            eachdict['surveyId'] = eachdict['objectId']
            eachdict['alerceId'] = eachdict['objectId']
        return messages

    @classmethod
    def canParseFile(self, messages):
        for eachdict in messages:
            if eachdict['publisher'] == self.source:
                return True

class ParserZTF(ParserGeneric):
    source = "ZTF"

    @classmethod
    def parse(self, messages):
        for eachdict in messages:
            eachdict['surveyId'] = eachdict['objectId']
            eachdict['alerceId'] = GeneratorID.generate_aid(eachdict['ra'],eachdict['dec'])
            eachdict['mjd'] = eachdict["jd"] - 2400000.5
        return messages

    @classmethod
    def canParseFile(self, messages):
        for eachdict in messages:
            if eachdict['publisher'] == self.source:
                return True

class GenericSaveStep(GenericStep):
    """GenericSaveStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """
    parser = ParserSelector()

    def __init__(self, consumer = None, config = None, level = logging.INFO, producer=None, db_connection=None,
                 **step_args):
        super().__init__(consumer,config=config, level=level)
        
        if "CLASS" in config["PRODUCER_CONFIG"]:
            Producer = get_class(config["PRODUCER_CONFIG"]["CLASS"])
        else:
            Producer = KafkaProducer

        self.producer = producer# or Producer(config["PRODUCER_CONFIG"])
        self.driver = db_connection or new_DBConnection(MongoDatabaseCreator)
        self.driver.connect(config["DB_CONFIG"])
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.logger.info(f"SAVE {self.version}")
        self.parser.registerParser(ParserATLAS)

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
        #TODO añadir cutoutTemplate a atlas alerts
        for k in ["cutoutDifference", "cutoutScience"]:#, "cutoutTemplate"]:
            del alerts[k]
    
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

    def get_last_alert(self, alerts):
        last_alert = alerts.candid.values.argmax()
        filtered_alerts = alerts.loc[
                          :, ["oid", "ndethist", "ncovhist", "jdstarthist", "jdendhist"]
                          ]
        last_alert = filtered_alerts.iloc[last_alert]
        return last_alert

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

    def preprocess_detections(self, detections):
        oids = detections.oid.values
        detectionsDB = self.get_detections(oids)

        index_detections = pd.MultiIndex.from_frame(
            detections[["oid", "candid"]]
        )
        index_light_curve_detections = pd.MultiIndex.from_frame(
            detectionsDB[["oid", "candid"]]
        )

        all_detections = pd.concat(
            [detectionsDB, detections], ignore_index=True
        )

        return all_detections

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} alerts")

        # Casting to a dataframe
        self.logger.info(f"Preprocessing alerts")

        self.parser.parse(messages)
        alerts = pd.DataFrame(messages)
        alerts.drop_duplicates("candid", inplace=True)
        alerts.reset_index(inplace=True)
        # Removing stamps
        self.remove_stamps(alerts)

        # Getting just the detections
        self.logger.info(f"Doing correction to detections")
        detections = pd.DataFrame(list(alerts["candidate"]))
        detections.loc[:, "sid"] = alerts["objectId"]
        detections.loc[:, "aid"] = alerts["alerceId"]

        # Getting data from database, and processing prv_candidates
        self.logger.info(f"Processing light curves")
        all_detections = self.preprocess_detections(detections)

        # Getting other tables
        objects = self.get_objects(detections["oid"].unique())
        objects = self.preprocess_objects(objects, all_detections, detections)

        # Insert new objects and update old objects
        self.insert_objects(objects)
        new_detections = all_detections["new"]
        self.insert_detections(all_detections.loc[new_detections])

        #self.produce(alerts, all_detections)

        del alerts
        del detections