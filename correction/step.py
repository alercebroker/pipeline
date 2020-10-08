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

from sqlalchemy.sql.expression import bindparam


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
PS1_KEYS = ["oid", "candid", "nmtchps"]
for i in range(1, 4):
    PS1_KEYS = PS1_KEYS + [f"{key}{i}" for key in PS1_MultKey]
SS_KEYS = [
    "oid",
    "candid",
    "ssdistnr",
    "ssmagnr",
    "ssnamenr"
]
GAIA_KEYS = [
    "oid",
    "candid",
    "neargaia",
    "neargaiabright",
    "maggaia",
    "maggaiabright"
]
REFERENCE_KEYS = [
    "oid",
    "rfid",
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
    "mjdstartref",
    "mjdendref"
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
OBJECT_UPDATE_PARAMS = [
    "ndethist",
    "ncovhist",
    "mjdstarthist",
    "mjdendhist",
    "corrected",
    # "stellar",
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

OBJECT_UPDATE_PARAMS_STMT = dict(zip(OBJECT_UPDATE_PARAMS,map(bindparam,OBJECT_UPDATE_PARAMS)))

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

    def get_magstats(self, oids):
        query = self.driver.query(MagStats).filter(MagStats.oid.in_(oids))
        return pd.read_sql(query.statement, self.driver.engine)

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
            if len(prv_detections) > 0:
                prv_detections = self.do_correction(prv_detections)

                #   Getting columns
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

            if len(prv_non_detections) > 0:
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

        if len(detections_old) > 0:
            self.update_old_objects(detections_old,alerts[~new_alerts])

    def process_ps1(self, metadata, detections):
        oids = metadata.oid.unique()
        new_metadata = ~detections.oid.isin(oids)
        new_values = detections.loc[new_metadata,detections.columns.isin(PS1_KEYS)]
        old_values = detections.loc[~new_metadata,detections.columns.isin(PS1_KEYS)]

        if len(new_values) > 0:
            self.logger.info(f"Inserting {len(new_values)} PS1 Metadata")
            new_values = new_values.where(pd.notnull(new_values), None)
            for i in range(1,4):
                new_values[f"unique{i}"] = True

            dict_new_values = new_values.to_dict('records')
            self.driver.query().bulk_insert(dict_new_values, Ps1_ztf)
        if len(old_values) > 0:
            self.logger.info(f"{len(old_values)} PS1 Metadata to be checked")
            join_metadata = old_values.join(metadata.set_index("oid"), on="oid", rsuffix="_old")
            for i in range(1,4):
                updates = []
                difference = join_metadata[
                                (join_metadata[f"objectidps{i}"] != join_metadata[f"objectidps{i}_old"]) & join_metadata[f"unique{i}"]
                            ]
                for oid in difference.oid:
                    updates.append({"_oid": oid, f"unique{i}": False})
                if len(updates) > 0:
                    self.logger.info(f"Updating {len(updates)} PS1 unique{i} metadata")
                    stmt = Ps1_ztf.__table__.update().\
                            where(Ps1_ztf.oid == bindparam('_oid')).\
                            values({f"unique{i}": bindparam(f'unique{i}')})
                    self.driver.engine.execute(stmt, updates)

        return pd.concat([metadata, new_values])

    def process_ss(self, metadata, detections):
        oids = metadata.oid.unique()
        new_metadata = ~detections.oid.isin(oids)
        new_values = detections.loc[new_metadata,detections.columns.isin(SS_KEYS)]
        if len(new_values) > 0:
            self.logger.info(f"Inserting {len(new_values)} Solar System Metadata")
            new_values = new_values.where(pd.notnull(new_values), None)
            dict_new_values = new_values.to_dict('records')
            self.driver.query().bulk_insert(dict_new_values, Ss_ztf)
        return pd.concat([metadata, new_values])


    def process_reference(self,metadata, detections):
        oids = metadata.oid.unique()
        index_metadata = pd.MultiIndex.from_frame(metadata[["oid", "rfid"]])
        index_detections = pd.MultiIndex.from_frame(detections[["oid", "rfid"]])
        already_on_db = index_detections.isin(index_metadata)
        detections["mjdstartref"] = detections["jdstartref"] - 2400000.5
        detections["mjdendref"] = detections["jdendref"] - 2400000.5
        new_values = detections.loc[~already_on_db, detections.columns.isin(REFERENCE_KEYS)]
        if len(new_values) > 0:
            self.logger.info(f"Inserting {len(new_values)} References")
            new_values = new_values.where(pd.notnull(new_values), None)
            dict_new_values = new_values.to_dict('records')
            self.driver.query().bulk_insert(dict_new_values, Reference)

        return pd.concat([metadata, new_values])

    def process_gaia(self, metadata, detections):
        oids = metadata.oid.unique()
        new_metadata = ~detections.oid.isin(oids)
        new_values = detections.loc[new_metadata,detections.columns.isin(GAIA_KEYS)]
        old_values = detections.loc[~new_metadata,detections.columns.isin(GAIA_KEYS)]

        if len(new_values) > 0:
            self.logger.info(f"Inserting {len(new_values)} Gaia Metadata")
            new_values = new_values.where(pd.notnull(new_values), None)
            new_values[f"unique1"] = True
            dict_new_values = new_values.to_dict('records')
            self.driver.query().bulk_insert(dict_new_values, Gaia_ztf)

        if len(old_values) > 0:
            self.logger.info(f"{len(old_values)} Gaia Metadata to be checked")
            join_metadata = old_values.join(metadata.set_index("oid"), on="oid", rsuffix="_old")
            updates = []
            difference = join_metadata[
                            ~np.isclose(join_metadata["neargaia"], join_metadata[f"neargaia_old"]) &\
                            ~np.isclose(join_metadata["neargaiabright"], join_metadata[f"neargaiabright_old"]) &\
                            ~np.isclose(join_metadata["maggaia"], join_metadata[f"maggaia_old"]) &\
                            ~np.isclose(join_metadata["maggaiabright"], join_metadata[f"maggaiabright_old"]) &\
                             join_metadata[f"unique1"]
                        ]
            for oid in difference.oid:
                updates.append({"_oid": oid, f"unique1": False})
            if len(updates) > 0:
                self.logger.info(f"Updating {len(updates)} Gaia unique1 metadata")
                stmt = Ps1_ztf.__table__.update().\
                        where(Ps1_ztf.oid == bindparam('_oid')).\
                        values({f"unique1": bindparam(f'unique1')})
                self.driver.engine.execute(stmt, updates)


        return pd.concat([metadata, new_values])


    def process_metadata(self, metadata, detections):
        metadata["ps1_ztf"] = self.process_ps1(metadata["ps1_ztf"], detections)
        metadata["ss_ztf"] = self.process_ss(metadata["ss_ztf"], detections)
        metadata["reference"] = self.process_reference(metadata["reference"], detections)
        metadata["gaia"] = self.process_gaia(metadata["gaia"], detections)
        return metadata

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

    def get_last_alert(self, alerts):
        last_alert = alerts.candid.values.argmax()
        filtered_alerts = alerts.loc[:,["oid","ndethist","ncovhist","jdstarthist","jdendhist"]]
        return filtered_alerts.iloc[last_alert]

    def get_object_data(self, detections):
        first_detection_position = detections.candid.values.argmin()
        first_detection = detections.iloc[first_detection_position]
        firstmjd = detections.mjd.min()
        lastmjd = detections.mjd.max()
        g_band_mag = detections[detections.fid == 1]['magpsf'].values
        r_band_mag = detections[detections.fid == 2]['magpsf'].values
        g_band_mag_corr = detections[detections.fid == 1]['magpsf_corr'].values
        r_band_mag_corr = detections[detections.fid == 2]['magpsf_corr'].values

        g_r_max, g_r_mean = self.get_colors(g_band_mag, r_band_mag)
        g_r_max_corr, g_r_mean_corr = self.get_colors(g_band_mag_corr, r_band_mag_corr)

        new_object = {
            "ndethist": int(first_detection["ndethist"]),
            "ncovhist": int(first_detection["ncovhist"]),
            "mjdstarthist": float(first_detection["jdstarthist"] - 2400000.5 ),
            "mjdendhist": float(first_detection["jdendhist"] - 2400000.5),
            "corrected": detections["corrected"].all(),
            "ndet": len(detections),
            "g_r_max": g_r_max,
            "g_r_max_corr": g_r_mean,
            "g_r_mean": g_r_max_corr,
            "g_r_mean_corr": g_r_mean_corr,
            "meanra": float(detections["ra"].mean()),
            "meandec": float(detections["dec"].mean()),
            "sigmara": float(detections["ra"].std()),
            "sigmadec": float(detections["dec"].std()),
            "deltajd": float(lastmjd - firstmjd),
            "firstmjd": float(firstmjd),
            "lastmjd": float(lastmjd),
            # "stellar":
            "step_id_corr": self.version
            }
        return pd.Series(new_object)

    def get_dataquality(self, candids):
        query = self.driver.query(Dataquality).filter(Dataquality.candid.in_(candids))
        return pd.read_sql(query.statement, self.driver.engine)

    def preprocess_dataquality(self, detections):
        dataquality = detections.loc[:,detections.columns.isin(DATAQUALITY_KEYS)]
        return dataquality

    def insert_new_objects(self, detections, alerts):
        oids = alerts.oid.unique()
        self.logger.info(f"Inserting {len(oids)} new objects")
        apply_last_alert = lambda x: self.get_last_alert(x)
        last_alerts = alerts.groupby('oid', sort=False).apply(apply_last_alert)
        detections_last_alert = detections.join(last_alerts, on="oid", rsuffix="alert")
        apply_get_object_data = lambda x: self.get_object_data(x)
        new_objects = detections_last_alert.groupby('oid', sort=False).apply(apply_get_object_data)
        new_objects.reset_index(inplace=True)
        dict_new_objects = new_objects.to_dict('records')
        self.driver.query().bulk_insert(dict_new_objects, Object)

    def insert_detections(self, detections):
        self.logger.info(f"Inserting {len(detections)} new detections")
        detections = detections.where(pd.notnull(detections), None)
        dict_detections = detections.to_dict('records')
        self.driver.query().bulk_insert(dict_detections, Detection)

    def insert_non_detections(self,non_detections):
        self.logger.info(f"Inserting {len(non_detections)} new non_detections")
        non_detections = non_detections.where(pd.notnull(non_detections), None)
        dict_non_detections = non_detections.to_dict('records')
        self.driver.query().bulk_insert(dict_non_detections, NonDetection)

    def insert_dataquality(self, dataquality):
        # Not inserting twice
        old_dataquality = self.get_dataquality(dataquality.candid.unique().tolist())
        already_on_db = dataquality.candid.isin(old_dataquality.candid)

        dataquality = dataquality[~already_on_db]
        self.logger.info(f"Inserting {len(dataquality)} new dataquality")

        dataquality = dataquality.where(pd.notnull(dataquality), None)
        dict_dataquality = dataquality.to_dict('records')
        self.driver.query().bulk_insert(dict_dataquality, Dataquality)

    def get_first_corrected(self, df):
        min_candid = df.candid.values.argmin()
        first_corr = df.corrected.iloc[min_candid]
        return first_corr


    def update_dubious(self,df):
        min_corr = df.groupby(["oid", "fid"], sort=False).apply(self.get_first_corrected)
        min_corr.name = "first_corrected"
        df = df.join(min_corr, on=["oid", "fid"])
        df.loc[:,"dubious"] = is_dubious(df.corrected, df.isdiffpos, df.first_corrected)
        df.drop(columns=["first_corrected"], inplace=True)
        return df

    def update_old_objects(self, detections, alerts):
        oids = alerts.oid.unique()
        self.logger.info(f"Inserting {len(oids)} new objects")

        # Getting last alert
        apply_last_alert = lambda x: self.get_last_alert(x)
        last_alerts = alerts.groupby('oid', sort=False).apply(apply_last_alert)
        # Joining with detections
        detections_last_alert = detections.join(last_alerts, on="oid", rsuffix="alert")
        apply_get_object_data = lambda x: self.get_object_data(x)

        # Getting detections info
        new_data = detections_last_alert.groupby('oid', sort=False).apply(apply_get_object_data)
        new_data.reset_index(inplace=True)

        # Renaming column for insert
        new_data.rename(columns={"oid":"_oid"}, inplace=True)

        dict_new_data = new_data.to_dict('records')
        stmt = Object.__table__.update().\
                where(Object.oid == bindparam('_oid')).\
                values(OBJECT_UPDATE_PARAMS_STMT)
        self.driver.engine.execute(stmt, dict_new_data)

    # def do_magstats(self, light_curves, metadata):
    #     detections = light_curves["detections"]
    #     non_detections = light_curves["non_detections"]
    #     ps1 = metadata["ps1_ztf"][["oid","distpsnr1","sgscore1"]]
    #     ps1.set_index("oid",inplace=True)
    #     ref = metadata["reference"][["oid", "rfid", "mjdstartref", "chinr","sharpnr"]]
    #     ref.set_index(["oid","rfid"],inplace=True)
    #     unique_ref = ref.sort_values('mjdstartref', ascending=False).drop_duplicates(["oid", "rfid"])
    #     det_ps1 = detections.join(ps1, on="oid", rsuffix="ps1")
    #     det_ps1_ref = det_ps1.join(unique_ref, on=["oid","rfid"], rsuffix="ref")
    #     magstats = det_ps1_ref.groupby(["oid","fid"], sort=False).apply(apply_mag_stats)
    #     return magstats

    def _produce(xmatch, metadata, light_curve):
        pass

    def produce(self, alerts, light_curves, metadata):
        # oids = alerts.oid.unique()
        # alerts.set_index("oid",inplace=True)
        # for oid in oids:
        #     oid_metdata = {
        #         "ps1_ztf": metadata["ps1_ztf"][metadata["ps1_ztf"].oid == oid],
        #         "ss_ztf": metadata["ss_ztf"][metadata["ss_ztf"].oid == oid],
        #         "gaia": metadata["gaia"][metadata["gaia"].oid == oid],
        #         "reference": metadata["reference"][metadata["reference"].oid == oid]
        #     }
        #     self._produce(["xmatches"], )
        pass
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
        # Index was changed to candid in do_correction
        detections.reset_index(inplace=True)

        new_dataquality = self.preprocess_dataquality(detections)

        # Getting data from database, and processing prv_candidates
        light_curves = self.process_lightcurves(corrected, alerts)

        # Update dubious
        light_curves["detections"] = self.update_dubious(light_curves["detections"])

        # Getting other tables
        objects = self.get_objects(corrected["oid"].unique())
        metadata = self.get_metadata(corrected["oid"].unique())
        magstats = self.get_magstats(corrected["oid"].unique())

        # Insert new objects and update old objects
        self.process_objects(objects, light_curves, corrected)
        new_detections = light_curves["detections"]["new"]
        self.insert_detections(light_curves["detections"].loc[new_detections])
        self.insert_dataquality(new_dataquality)
        new_non_detections = light_curves["non_detections"]["new"]
        self.insert_non_detections(light_curves["non_detections"].loc[new_non_detections])
        metadata = self.process_metadata(metadata, detections)
        # new_magstats = self.do_magstats(light_curves, metadata)

        self.produce(alerts, light_curves, metadata)

        del alerts
        del detections
        del corrected
        del light_curves
