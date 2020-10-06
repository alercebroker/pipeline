from apf.core import get_class
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from db_plugins.db.sql.models import (
    Object,
    Detection,
    NonDetection,
    Ps1_ztf,
    Ss_ztf,
    Reference,
    MagStats,
    Dataquality,
    Gaia_ztf,
    Step,
)
from db_plugins.db.sql.serializers import (
    Gaia_ztfSchema,
    Ss_ztfSchema,
    Ps1_ztfSchema,
    ReferenceSchema,
)
from db_plugins.db.sql import SQLConnection
from lc_correction.compute import (
    apply_correction_df,
    is_dubious,
    apply_mag_stats,
    do_dmdt,
    DISTANCE_THRESHOLD,
)
from astropy.time import Time
from pandas import DataFrame, Series, concat
import pandas as pd

import numpy as np
import logging
import numbers
import datetime


logging.getLogger("GP").setLevel(logging.WARNING)
np.seterr(divide="ignore")

DET_KEYS = [
    "oid",
    "candid",
    "mjd",
    "fid",
    "pid",
    "diffmaglim",
    "isdiffpos",
    "nid",
    "ra",
    "dec",
    "magpsf",
    "sigmapsf",
    "magap",
    "sigmagap",
    "distnr",
    "rb",
    "rbversion",
    "drb",
    "drbversion",
    "magapbig",
    "sigmagapbig",
    "rfid",
    "magpsf_corr",
    "sigmapsf_corr",
    "sigmapsf_corr_ext",
    "corrected",
    "dubious",
    "parent_candid",
    "has_stamp",
    "step_id_corr",
]
NON_DET_KEYS = ["oid", "mjd", "diffmaglim", "fid"]
COR_KEYS = ["magpsf_corr", "sigmapsf_corr", "sigmapsf_corr_ext"]
PS1_MultKey = ["objectidps", "sgmag", "srmag", "simag", "szmag", "sgscore", "distpsnr"]
PS1_KEYS = ["candid", "nmtchps"]
for i in range(1, 4):
    PS1_KEYS = PS1_KEYS + [f"{key}{i}" for key in PS1_MultKey]
REFERENCE_KEYS = [
    "candid",
    "fid",
    "rcid",
    "field",
    "magnr",
    "sigmagnr",
    "chinr",
    "sharpnr",
    "chinr",
    "ranr",
    "decnr",
    "nframesref",
]
DATAQUALITY_KEYS = [
    "oid",
    "candid",
    "fid",
    "xpos",
    "ypos",
    "chipsf",
    "sky",
    "fwhm",
    "classtar",
    "mindtoedge",
    "seeratio",
    "aimage",
    "bimage",
    "aimagerat",
    "bimagerat",
    "nneg",
    "nbad",
    "sumrat",
    "scorr",
    "magzpsci",
    "magzpsciunc",
    "magzpscirms",
    "clrcoeff",
    "clrcounc",
    "dsnrms",
    "ssnrms",
    "nmatches",
    "zpclrcov",
    "zpmed",
    "clrmed",
    "clrrms",
    "exptime",
]


class Correction(GenericStep):
    """Correction Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(self, consumer=None, level=logging.INFO, config=None, **step_args):
        super().__init__(consumer, level=level, config=config, **step_args)

        if "CLASS" in config["PRODUCER_CONFIG"]:
            Producer = get_class(config["PRODUCER_CONFIG"]["CLASS"])
        else:
            Producer = KafkaProducer

        self.producer = Producer(config["PRODUCER_CONFIG"])
        self.driver = SQLConnection()
        self.driver.connect(config["DB_CONFIG"]["SQL"])
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.logger.info(f"CORRECTION {self.version}")
        # Storing step_id
        self.driver.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_ID"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["STEP_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )
        self.driver.session.commit()

    def preprocess_detections(self, detections, is_prv_candidate=False) -> None:
        detections.loc[:,"mjd"] = detections["jd"] - 2400000.5
        detections.loc[:,"has_stamp"] = True
        detections.loc[:,"step_id_corr"] = self.version
        return detections

    def remove_stamps(self, alerts):
        for k in ["cutoutDifference", "cutoutScience", "cutoutTemplate"]:
            del alerts[k]

    def do_correction(self, detections, inplace=False) -> dict:
        fid = detections.fid.values
        candid = detections.candid.values
        corrected = apply_correction_df(detections)
        corrected.reset_index(inplace=True)

        corrected.loc[:, "fid"] = fid
        corrected.loc[:, "candid"] = candid
        return corrected

    def get_objects(self, oids):
        query = self.driver.query(Object).filter(Object.oid.in_(oids))
        return pd.read_sql(query.statement,self.driver.engine)

    def get_detections(self,oids):
        query = self.driver.query(Detection).filter(Detection.oid.in_(oids))
        return pd.read_sql(query.statement,self.driver.engine)

    def get_non_detections(self,oids):
        query = self.driver.query(NonDetection).filter(NonDetection.oid.in_(oids))
        return pd.read_sql(query.statement,self.driver.engine)

    def get_lightcurves(self, oids):
        light_curves = {}
        light_curves["detections"] = self.get_detections(oids)
        light_curves["non_detections"] = self.get_non_detections(oids)
        return light_curves

    def get_ps1(self, oids):
        query = self.driver.query(Ps1_ztf).filter(Ps1_ztf.oid.in_(oids))
        return pd.read_sql(query.statement, self.driver.engine)

    def get_ss(self, oids):
        query = self.driver.query(Ss_ztf).filter(Ss_ztf.oid.in_(oids))
        return pd.read_sql(query.statement, self.driver.engine)

    def get_reference(self, oids):
        query = self.driver.query(Reference).filter(Reference.oid.in_(oids))
        return pd.read_sql(query.statement, self.driver.engine)

    def get_gaia(self, oids):
        query = self.driver.query(Gaia_ztf).filter(Gaia_ztf.oid.in_(oids))
        return pd.read_sql(query.statement, self.driver.engine)

    def get_metadata(self, oids):
        metadata = {}
        metadata["ps1_ztf"] = self.get_ps1(oids)
        metadata["ss_ztf"] = self.get_ss(oids)
        metadata["reference"] = self.get_reference(oids)
        metadata["gaia"] = self.get_gaia(oids)
        return metadata

    def get_magstats(self, light_curves, metadata):
        pass

    def cast_non_detection(self, object_id: str, candidate: dict) -> dict:
        non_detection = {
            "oid": object_id,
            "mjd": candidate["mjd"],
            "diffmaglim": candidate["diffmaglim"],
            "fid": candidate["fid"],
        }
        return non_detection

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
                candidate["mjd"] = candidate["jd"] - 2400000.5

            if is_non_detection:
                non_detection = self.cast_non_detection(oid, candidate)
                non_detections.append(non_detection)
            else:
                candidate["parent_candid"] = candid
                candidate["has_stamp"] = False
                candidate["oid"] = oid
                candidate["objectId"] = oid # Used in correction
                candidate["step_id_corr"] = self.version
                detections.append(candidate)

        return pd.DataFrame(detections), pd.DataFrame(non_detections)

    def process_lightcurves(self, detections, alerts):
        oids = detections.oid.values

        detections.loc[:,"parent_candid"] = None
        detections.loc[:, "has_stamp"] = True
        filtered_detections = detections.loc[:, DET_KEYS]
        filtered_detections.loc[:,"new"] = True

        light_curves = self.get_lightcurves(oids)
        light_curves["detections"]["new"] = False
        light_curves["non_detections"]["new"] = False

        # Removing already on db, similar to drop duplicates
        index_detections = pd.MultiIndex.from_frame(filtered_detections[["oid", "candid"]])
        index_light_curve_detections = pd.MultiIndex.from_frame(
                                    light_curves["detections"][["oid", "candid"]]
                                )
        already_on_db = index_detections.isin(index_light_curve_detections)
        filtered_detections = filtered_detections[~already_on_db]
        light_curves["detections"] = pd.concat([light_curves["detections"], filtered_detections])


        prv_detections = []
        prv_non_detections = []
        for _,alert in alerts.iterrows():
            if "prv_candidates" in alert:
                alert_prv_detections, alert_prv_non_detections = self.get_prv_candidates(alert)
                prv_detections.append(alert_prv_detections)
                prv_non_detections.append(alert_prv_non_detections)
        prv_detections = pd.concat(prv_detections)
        prv_non_detections = pd.concat(prv_non_detections)

        if len(prv_detections) > 0:
            prv_detections.drop_duplicates(["oid","candid"], inplace=True)
            # Checking if already on the database
            index_prv_detections = pd.MultiIndex.from_frame(prv_detections[["oid", "candid"]])
            already_on_db = index_prv_detections.isin(index_light_curve_detections)
            prv_detections = prv_detections[~already_on_db]

            # Doing correction
            prv_detections = self.do_correction(prv_detections)

            #Getting columns
            current_keys = [key for key in DET_KEYS if key in prv_detections.columns]
            prv_detections = prv_detections.loc[:, current_keys]
            prv_detections.loc[:,"new"] = True
            light_curves["detections"] = pd.concat([light_curves["detections"], prv_detections])

        if len(prv_non_detections) > 0:
            # Using round 5 to have 5 decimals of precision
            prv_non_detections.loc[:, "round_mjd"] = prv_non_detections["mjd"].round(5)
            light_curves["non_detections"].loc[:, "round_mjd"] = light_curves["non_detections"]["mjd"].round(5)

            prv_non_detections.drop_duplicates(["oid","fid", "round_mjd"])

            # Checking if already on the database
            index_prv_non_detections = pd.MultiIndex.from_frame(prv_non_detections[["oid", "fid", "round_mjd"]])
            index_light_curve_non_detections = pd.MultiIndex.from_frame(
                                                light_curves["non_detections"][["oid", "fid", "round_mjd"]]
                                            )
            already_on_db = index_prv_non_detections.isin(index_light_curve_non_detections)
            prv_non_detections = prv_non_detections[~already_on_db]

            # Dropping auxiliary column
            light_curves["non_detections"].drop(columns=["round_mjd"], inplace=True)
            prv_non_detections.drop(columns=["round_mjd"], inplace=True)
            prv_non_detections.loc[:, "new"] = True
            light_curves["non_detections"] = pd.concat([light_curves["non_detections"], prv_non_detections])

        return light_curves


    def process_objects(self, objects, light_curves, alerts):

        new_objects = ~light_curves["detections"]["oid"].isin(objects.oid)
        new_alerts = ~alerts["oid"].isin(objects.oid)

        detections_new =  light_curves["detections"][new_objects]
        detections_old = light_curves["detections"][~new_objects]

        if len(detections_new) > 0:
            self.insert_new_objects(detections_new, alerts[new_alerts])

        #if len(detections_old) > 0:
        # self.update_old_objects(detections_old, objects)

    def get_colors(self, g_band, r_band):
        g_max = g_band.min() if len(g_band) > 0 else np.nan
        r_max = r_band.min() if len(r_band) > 0 else np.nan
        g_mean = g_band.mean() if len(g_band) > 0 else np.nan
        r_mean = r_band.mean() if len(r_band) > 0 else np.nan

        g_r_max = g_max - r_max
        g_r_mean = g_mean - r_mean

        g_r_max = float(g_r_max) if not np.isnan(g_r_max) else None
        g_r_mean = float(g_r_mean) if not np.isnan(g_r_mean) else None
        return g_r_max, g_r_mean


    def get_object_data(self,oid, detections, alerts):
        mjd_detections = detections.sort_values("mjd")
        mjd_alerts = alerts.sort_values("mjd")

        first_alert = mjd_alerts.iloc[0]
        first_detection = mjd_detections.iloc[0]
        firstmjd = mjd_detections["mjd"].min()
        lastmjd = mjd_detections["mjd"].max()
        g_band_mag = mjd_detections[mjd_detections.fid == 1]['magpsf'].values
        r_band_mag = mjd_detections[mjd_detections.fid == 2]['magpsf'].values
        g_band_mag_corr = mjd_detections[mjd_detections.fid == 1]['magpsf_corr'].values
        r_band_mag_corr = mjd_detections[mjd_detections.fid == 2]['magpsf_corr'].values

        g_r_max, g_r_mean = self.get_colors(g_band_mag, r_band_mag)
        g_r_max_corr, g_r_mean_corr = self.get_colors(g_band_mag_corr, r_band_mag_corr)

        new_object = {
            "oid": oid,
            "ndethist": int(first_alert["ndethist"]),
            "ncovhist": int(first_alert["ncovhist"]),
            "mjdstarthist": float(first_alert["jdstarthist"] - 2400000.5),
            "mjdendhist": float(first_alert["jdendhist"] - 2400000.5),
            "corrected": mjd_detections["corrected"].all(),
            "ndet": len(mjd_detections),
            "g-r_max": g_r_max,
            "g-r_max_corr": g_r_mean,
            "g-r_mean": g_r_max_corr,
            "g-r_mean_corr": g_r_mean_corr,
            "meanra": float(mjd_detections["ra"].mean()),
            "meandec": float(mjd_detections["dec"].mean()),
            "sigmara": float(mjd_detections["ra"].std()),
            "sigmadec": float(mjd_detections["dec"].std()),
            "deltajd": float(lastmjd - firstmjd),
            "firstmjd": float(firstmjd),
            "lastmjd": float(lastmjd),
            # "stellar":
            "step_id_corr": self.version
            }
        return new_object


    def insert_new_objects(self, detections, alerts):
        oids = alerts.oid.values
        self.logger.info(f"Inserting new objects ({len(oids)} objects)")
        new_objects = []
        for oid in oids:
            new_obj = self.get_object_data(oid, detections[detections.oid == oid], alerts[alerts.oid == oid])
            new_objects.append(new_obj)
        self.driver.query().bulk_insert(new_objects, Object)

    def insert_detections(self, detections):
        self.logger.info(f"Inserting {len(detections)} new detections")
        # detections.drop(columns=["new"], inplace=True)
        detections = detections.where(pd.notnull(detections), None)
        dict_detections = detections.to_dict('records')
        self.driver.query().bulk_insert(dict_detections, Detection)

    def insert_non_detections(self,non_detections):
        self.logger.info(f"Inserting {len(non_detections)} new non_detections")
        # non_detections.drop(columns=["new"], inplace=True)
        non_detections = non_detections.where(pd.notnull(non_detections), None)
        dict_non_detections = non_detections.to_dict('records')
        self.driver.query().bulk_insert(dict_non_detections, NonDetection)

    def execute(self, messages):
        # Casting to a dataframe
        alerts = pd.DataFrame(messages)

        # Removing stamps
        self.remove_stamps(alerts)

        # Getting just the detections
        detections = pd.DataFrame(list(alerts["candidate"]))
        detections.loc[:, "objectId"] = alerts["objectId"]
        detections.loc[:, "oid"] = alerts["objectId"]
        detections = self.preprocess_detections(detections)
        corrected = self.do_correction(detections)
        del detections

        light_curves = self.process_lightcurves(corrected, alerts)
        del alerts

        objects = self.get_objects(corrected["oid"].values)
        metadata = self.get_metadata(corrected["oid"].values)
        self.process_objects(objects, light_curves, corrected)
        del corrected

        new_detections = light_curves["detections"]["new"]
        self.insert_detections(light_curves["detections"].loc[new_detections])
        new_non_detections = light_curves["non_detections"]["new"]
        self.insert_non_detections(light_curves["non_detections"].loc[new_non_detections])
        del light_curves
