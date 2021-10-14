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
    NonDetection,
)
from db_plugins.db.mongo.connection import MongoDatabaseCreator

import numpy as np
import pandas as pd

OBJ_KEYS = ["aid", "sid", "oid", "lastmjd", "firstmjd", "meanra", "meandec", "sigmara", "sigmadec"]
DET_KEYS = ["aid", "sid", "oid", "candid", "mjd", "fid", "ra", "dec", "rb", "mag", "sigmag"]
NON_DET_KEYS = ["aid", "sid", "mjd", "diffmaglim", "fid", "extra_fields"]
COR_KEYS = ["magpsf_corr", "sigmapsf_corr", "sigmapsf_corr_ext"]

OBJECT_UPDATE_PARAMS = [
    "ndethist",
    "ncovhist",
    "mjdstarthist",
    "mjdendhist",
    "corrected",
    "stellar",
    "ndet",
    "g_r_max",
    "g_r_mean",
    "g_r_max_corr",
    "g_r_mean_corr",
    "meanra",
    "meandec",
    "sigmara",
    "sigmadec",
    "deltajd",
    "firstmjd",
    "lastmjd",
    "step_id_corr",
]

MAGSTATS_TRANSLATE = {
    "magpsf_mean": "magmean",
    "magpsf_median": "magmedian",
    "magpsf_max": "magmax",
    "magpsf_min": "magmin",
    "sigmapsf": "magsigma",
    "magpsf_last": "maglast",
    "magpsf_first": "magfirst",
    "magpsf_corr_mean": "magmean_corr",
    "magpsf_corr_median": "magmedian_corr",
    "magpsf_corr_max": "magmax_corr",
    "magpsf_corr_min": "magmin_corr",
    "sigmapsf_corr": "magsigma_corr",
    "magpsf_corr_last": "maglast_corr",
    "magpsf_corr_first": "magfirst_corr",
    "first_mjd": "firstmjd",
    "last_mjd": "lastmjd",
}

MAGSTATS_UPDATE_KEYS = [
    "stellar",
    "corrected",
    "ndet",
    "ndubious",
    "dmdt_first",
    "dm_first",
    "sigmadm_first",
    "dt_first",
    "magmean",
    "magmedian",
    "magmax",
    "magmin",
    "magsigma",
    "maglast",
    "magfirst",
    "magmean_corr",
    "magmedian_corr",
    "magmax_corr",
    "magmin_corr",
    "magsigma_corr",
    "maglast_corr",
    "magfirst_corr",
    "firstmjd",
    "lastmjd",
    "step_id_corr",
]

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
            eachdict['candidate']['mag'] = eachdict['candidate']['Mag']
            del eachdict['candidate']['Mag']
            eachdict['candidate']['sigmag'] = eachdict['candidate']['Dmag']
            del eachdict['candidate']['Dmag']
            eachdict['candidate']['aimage'] = eachdict['candidate']['Major']
            del eachdict['candidate']['Major']
            eachdict['candidate']['bimage'] = eachdict['candidate']['Minor']
            del eachdict['candidate']['Minor']
            eachdict['surveyId'] = self.source
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
            eachdict['surveyId'] = self.source
            eachdict['candidate']['mag'] = eachdict['candidate']['magpsf']
            del eachdict['candidate']['magpsf']
            eachdict['candidate']['sigmag'] = eachdict['candidate']['sigmapsf']
            del eachdict['candidate']['sigmapsf']
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
        objects = self.driver.query(Object).filter(model=Object,field="_id", op="in", values=oids)
        return pd.DataFrame(objects,columns = OBJ_KEYS)

    def get_detections(self, oids):
        detections = self.driver.query(Detection).filter(model=Detection,field="aid", op="in", values=oids)
        return pd.DataFrame(detections,columns = DET_KEYS)

    def get_non_detections(self, oids):
        non_detections = self.driver.query(NonDetection).filter(model=NonDetection, field="aid", op="in", values=oids)
        return pd.DataFrame(non_detections)

    def remove_stamps(self, alerts):
        #TODO añadir cutoutTemplate a atlas alerts
        for k in ["cutoutDifference", "cutoutScience"]:#, "cutoutTemplate"]:
            del alerts[k]
    
    def insert_objects(self, objects):
        new_objects = objects["new"]
        objects.drop_duplicates(["aid"], inplace=True)
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
            dict_to_update = to_update.to_dict("records")
            for object in dict_to_update:
                self.driver.query().update(object["aid"], object, Object)
            
    def insert_detections(self, detections):
        self.logger.info(f"Inserting {len(detections)} new detections")
        detections = detections.where(detections.notnull(), None)
        dict_detections = detections.to_dict("records")
        for object in dict_detections:
            del object['new']
        self.driver.query().bulk_insert(dict_detections, Detection)

    def apply_objstats_from_correction(self, df):
        response = {}
        df_mjd = df.mjd
        idxmin = df_mjd.values.argmin()
        df_min = df.iloc[idxmin]
        df_ra = df.ra
        df_dec = df.dec
        response["meanra"] = df_ra.mean()
        response["meandec"] = df_dec.mean()
        response["sigmara"] = df_ra.std(ddof=0)
        response["sigmadec"] = df_dec.std(ddof=0)
        response["firstmjd"] = df_mjd.min()
        response["lastmjd"] = df_mjd.max()
        response["sid"] = df_min.sid
        response["oid"] = df_min.oid
        return pd.Series(response)

    def get_last_alert(self, alerts):
        # TODO change mjd to candid
        last_alert = alerts.mjd.values.argmax()
        filtered_alerts = alerts.loc[
                          :, ["aid"]
                          ]
        last_alert = filtered_alerts.iloc[last_alert]
        return last_alert

    def preprocess_objects(self, objects, light_curves, alerts):
        oids = objects.aid.unique()
        apply_last_alert = lambda x: self.get_last_alert(x)
        last_alerts = alerts.groupby("aid", sort=False).apply(apply_last_alert)
        last_alerts.drop(columns=["aid"], inplace=True)

        detections = light_curves["detections"]
        detections_last_alert = detections.join(last_alerts, on="aid")
        detections_last_alert.drop_duplicates(["candid", "aid"], inplace=True)
        detections_last_alert.reset_index(inplace=True)

        new_objects = detections_last_alert.groupby("aid").apply(self.apply_objstats_from_correction)
        new_objects.reset_index(inplace=True)

        new_names = dict(
            [(col, col.replace("-", "_")) for col in new_objects.columns if "-" in col]
        )

        new_objects.rename(columns={**new_names}, inplace=True)
        new_objects["new"] = ~new_objects.aid.isin(oids)

        return new_objects

    def preprocess_detections(self, detections):
        oids = detections.aid.values
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

    def get_lightcurves(self, oids):
        light_curves = {}
        light_curves["detections"] = self.get_detections(oids)
        light_curves["non_detections"] = self.get_non_detections(oids)
        self.logger.info(
            f"Light Curves: {len(light_curves['detections'])} detections, {len(light_curves['non_detections'])} non_detections"
        )
        return light_curves

    def get_prv_candidates(self, alert):
        prv_candidates = alert["prv_candidates"]
        candid = alert["candid"]
        oid = alert["objectId"]

        detections = []
        non_detections = []

        if prv_candidates is None:
            return pd.DataFrame([]), pd.DataFrame([])
        else:
            for candidate in prv_candidates:
                is_non_detection = candidate["candid"] is None
                # TODO change to ZTF parser
                candidate["mjd"] = candidate["jd"] - 2400000.5

                if is_non_detection:
                    non_detection = self.cast_non_detection(oid, candidate)
                    non_detections.append(non_detection)
                else:
                    candidate["parent_candid"] = candid
                    candidate["has_stamp"] = False
                    candidate["oid"] = oid
                    candidate["sid"] = oid  # Used in correction
                    candidate["step_id_corr"] = self.version
                    detections.append(candidate)

        return detections, non_detections

    def preprocess_lightcurves(self, detections, alerts):

        oids = detections.aid.values

        detections.loc[:, "parent_candid"] = None
        detections.loc[:, "has_stamp"] = True
        filtered_detections = detections.loc[:, detections.columns.isin(DET_KEYS)]
        filtered_detections.loc[:, "new"] = True

        light_curves = self.get_lightcurves(oids)
        light_curves["detections"]["new"] = False
        light_curves["non_detections"]["new"] = False

        # Removing already on db, similar to drop duplicates
        index_detections = pd.MultiIndex.from_frame(
            filtered_detections[["aid", "candid"]]
        )
        index_light_curve_detections = pd.MultiIndex.from_frame(
            light_curves["detections"][["aid", "candid"]]
        )
        already_on_db = index_detections.isin(index_light_curve_detections)
        filtered_detections = filtered_detections[~already_on_db]
        light_curves["detections"] = pd.concat(
            [light_curves["detections"], filtered_detections], ignore_index=True
        )

        prv_detections = []
        prv_non_detections = []
        for _, alert in alerts.iterrows():
            if "prv_candidates" in alert:
                (
                    alert_prv_detections,
                    alert_prv_non_detections,
                ) = self.get_prv_candidates(alert)
                prv_detections.extend(alert_prv_detections)
                prv_non_detections.extend(alert_prv_non_detections)

        if len(prv_detections) > 0:
            prv_detections = pd.DataFrame(prv_detections)
            prv_detections.drop_duplicates(["oid", "candid"], inplace=True)
            # Checking if already on the database
            index_prv_detections = pd.MultiIndex.from_frame(
                prv_detections[["oid", "candid"]]
            )
            index_light_curve_detections = pd.MultiIndex.from_frame(
                light_curves["detections"][["oid", "candid"]]
            )
            already_on_db = index_prv_detections.isin(index_light_curve_detections)
            prv_detections = prv_detections[~already_on_db]

            # Doing correction
            if len(prv_detections) > 0:
                prv_detections["jdendref"] = np.nan
                prv_detections = self.do_correction(prv_detections)

                #   Getting columns
                current_keys = [
                    key for key in DET_KEYS if key in prv_detections.columns
                ]
                prv_detections = prv_detections.loc[:, current_keys]
                prv_detections.loc[:, "new"] = True
                light_curves["detections"] = pd.concat(
                    [light_curves["detections"], prv_detections], ignore_index=True
                )

        if len(prv_non_detections) > 0:
            prv_non_detections = pd.DataFrame(prv_non_detections)
            # Using round 5 to have 5 decimals of precision
            prv_non_detections.loc[:, "round_mjd"] = prv_non_detections["mjd"].round(5)
            light_curves["non_detections"].loc[:, "round_mjd"] = light_curves[
                "non_detections"
            ]["mjd"].round(5)

            prv_non_detections.drop_duplicates(["oid", "fid", "round_mjd"])

            # Checking if already on the database
            index_prv_non_detections = pd.MultiIndex.from_frame(
                prv_non_detections[["oid", "fid", "round_mjd"]]
            )
            index_light_curve_non_detections = pd.MultiIndex.from_frame(
                light_curves["non_detections"][["oid", "fid", "round_mjd"]]
            )
            already_on_db = index_prv_non_detections.isin(
                index_light_curve_non_detections
            )
            prv_non_detections = prv_non_detections[~already_on_db]
            prv_non_detections.drop_duplicates(inplace=True)

            if len(prv_non_detections) > 0:
                # Dropping auxiliary column
                light_curves["non_detections"].drop(columns=["round_mjd"], inplace=True)
                prv_non_detections.drop(columns=["round_mjd"], inplace=True)
                prv_non_detections.loc[:, "new"] = True
                light_curves["non_detections"] = pd.concat(
                    [light_curves["non_detections"], prv_non_detections],
                    ignore_index=True,
                )

        return light_curves

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
        detections.loc[:, "sid"] = alerts["surveyId"]
        detections.loc[:, "aid"] = alerts["alerceId"]
        detections.loc[:, "oid"] = alerts["objectId"]

        #corrected = self.do_correction(detections)
        # Index was changed to candid in do_correction
        detections.reset_index(inplace=True)

        # Getting data from database, and processing prv_candidates
        self.logger.info(f"Processing light curves")
        light_curves = self.preprocess_lightcurves(detections, alerts)

        # Getting other tables
        objects = self.get_objects(detections["aid"].unique())

        objects = self.preprocess_objects(objects, light_curves, detections)

        self.logger.info(f"Setting objects flags")
        # Setting flags in objects
        objects.set_index("aid", inplace=True)
        # objects.loc[obj_flags.index, "diffpos"] = obj_flags["diffpos"]
        # objects.loc[obj_flags.index, "reference_change"] = obj_flags["reference_change"]
        objects.reset_index(inplace=True)

        # Insert new objects and update old objects
        self.insert_objects(objects)
        new_detections = light_curves["detections"]["new"]
        self.insert_detections(light_curves["detections"].loc[new_detections])

        #self.produce(alerts, light_curves)

        del alerts
        del detections
        del light_curves