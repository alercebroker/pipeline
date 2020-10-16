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
    do_dmdt_df,
    apply_object_stats_df,
    DISTANCE_THRESHOLD,
)
from astropy.time import Time
from pandas import DataFrame, Series, concat
import pandas as pd

import numpy as np
import logging
import numbers
import datetime
import warnings

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
SS_KEYS = ["oid", "candid", "ssdistnr", "ssmagnr", "ssnamenr"]
GAIA_KEYS = ["oid", "candid", "neargaia", "neargaiabright", "maggaia", "maggaiabright"]
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
    "mjdendref",
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
MAGSTATS_UPDATE_KEYS_STMT = dict(
    zip(MAGSTATS_UPDATE_KEYS, map(bindparam, MAGSTATS_UPDATE_KEYS))
)
OBJECT_UPDATE_PARAMS_STMT = dict(
    zip(OBJECT_UPDATE_PARAMS, map(bindparam, OBJECT_UPDATE_PARAMS))
)

MIN_DETECTIONS_TO_PRODUCE = 6


class Correction(GenericStep):
    """Correction Description

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
        level=logging.INFO,
        config=None,
        db_connection=None,
        producer=None,
        **step_args,
    ):
        super().__init__(consumer, level=level, config=config, **step_args)

        if "CLASS" in config["PRODUCER_CONFIG"]:
            Producer = get_class(config["PRODUCER_CONFIG"]["CLASS"])
        else:
            Producer = KafkaProducer

        self.producer = producer or Producer(config["PRODUCER_CONFIG"])
        self.driver = db_connection or SQLConnection()
        self.driver.connect(config["DB_CONFIG"]["SQL"])
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.logger.info(f"CORRECTION {self.version}")
        if not step_args.get("test_mode", False):
            self.insert_step_metadata()

    def insert_step_metadata(self):
        self.driver.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_ID"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["STEP_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )
        self.driver.session.commit()

    def remove_stamps(self, alerts):
        for k in ["cutoutDifference", "cutoutScience", "cutoutTemplate"]:
            del alerts[k]

    def preprocess_metadata(self, metadata, detections):
        metadata["ps1_ztf"] = self.preprocess_ps1(metadata["ps1_ztf"], detections)
        metadata["ss_ztf"] = self.preprocess_ss(metadata["ss_ztf"], detections)
        metadata["reference"] = self.preprocess_reference(
            metadata["reference"], detections
        )
        metadata["gaia"] = self.preprocess_gaia(metadata["gaia"], detections)
        return metadata

    def preprocess_objects(self, objects, light_curves, alerts, magstats):
        oids = objects.oid.unique()
        apply_last_alert = lambda x: self.get_last_alert(x)
        last_alerts = alerts.groupby("oid", sort=False).apply(apply_last_alert)

        detections = light_curves["detections"]
        detections_last_alert = detections.join(last_alerts, on="oid", rsuffix="alert")
        detections_last_alert["objectId"] = detections_last_alert.oid
        detections_last_alert.drop_duplicates("candid", inplace=True)
        detections_last_alert.reset_index(inplace=True)
        magstats["objectId"] = magstats.oid

        new_objects = apply_object_stats_df(
            detections_last_alert, magstats, step_name=self.version
        )
        new_objects.reset_index(inplace=True)
        
        new_names = dict(
            [(col, col.replace("-", "_")) for col in new_objects.columns if "-" in col]
        )

        new_objects.rename(columns={"objectId": "oid", **new_names}, inplace=True)
        new_objects["new"] = ~new_objects.oid.isin(oids)
        new_objects["deltajd"] = new_objects["deltamjd"]

        detections_last_alert.drop(columns=["objectId"], inplace=True)
        magstats.drop(columns=["objectId"], inplace=True)

        return new_objects

    def preprocess_detections(self, detections, is_prv_candidate=False) -> None:
        detections.loc[:, "mjd"] = detections["jd"] - 2400000.5
        detections.loc[:, "has_stamp"] = True
        detections.loc[:, "step_id_corr"] = self.version
        return detections

    def preprocess_dataquality(self, detections):
        dataquality = detections.loc[:, detections.columns.isin(DATAQUALITY_KEYS)]
        return dataquality

    def do_correction(self, detections, inplace=False) -> dict:
        fid = detections.fid.values
        candid = detections.candid.values
        corrected = apply_correction_df(detections)
        corrected.reset_index(inplace=True)
        corrected.loc[:, "fid"] = fid
        corrected.loc[:, "candid"] = candid
        return corrected

    def do_dubious(self, df):
        min_corr = df.groupby(["oid", "fid"], sort=False).apply(
            self.get_first_corrected
        )
        min_corr.name = "first_corrected"
        df = df.join(min_corr, on=["oid", "fid"])
        df.loc[:, "dubious"] = is_dubious(
            df.corrected, df.isdiffpos, df.first_corrected
        )
        df.drop(columns=["first_corrected"], inplace=True)
        return df

    def do_magstats(self, light_curves, metadata, magstats):
        magstats_index = pd.MultiIndex.from_frame(magstats[["oid", "fid"]])
        detections = light_curves["detections"]
        non_detections = light_curves["non_detections"]
        ps1 = metadata["ps1_ztf"][["oid", "distpsnr1", "sgscore1"]]
        ps1.set_index("oid", inplace=True)
        ref = metadata["reference"][["oid", "rfid", "chinr", "sharpnr"]]
        ref.set_index(["oid", "rfid"], inplace=True)
        det_ps1 = detections.join(ps1, on="oid", rsuffix="ps1")
        det_ps1_ref = det_ps1.join(ref, on=["oid", "rfid"], rsuffix="ref")
        det_ps1_ref.reset_index(inplace=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            new_magstats = det_ps1_ref.groupby(["oid", "fid"], sort=False).apply(
                apply_mag_stats
            )
        new_magstats.reset_index(inplace=True)
        new_magstats_index = pd.MultiIndex.from_frame(new_magstats[["oid", "fid"]])
        new_magstats["new"] = ~new_magstats_index.isin(magstats_index)
        new_magstats["step_id_corr"] = self.version
        new_magstats.drop_duplicates(["oid", "fid"], inplace=True)

        return new_magstats

    def do_dmdt(self, light_curves, magstats):
        non_detections = light_curves["non_detections"]
        non_detections["objectId"] = non_detections["oid"]
        magstats["objectId"] = magstats["oid"]
        dmdt = do_dmdt_df(magstats, non_detections)
        non_detections.drop(columns=["objectId"], inplace=True)
        magstats.drop(columns=["objectId"], inplace=True)
        return dmdt

    def get_objects(self, oids):
        query = self.driver.query(Object).filter(Object.oid.in_(oids))
        return pd.read_sql(query.statement, self.driver.engine)

    def get_detections(self, oids):
        query = self.driver.query(Detection).filter(Detection.oid.in_(oids))
        return pd.read_sql(query.statement, self.driver.engine)

    def get_non_detections(self, oids):
        query = self.driver.query(NonDetection).filter(NonDetection.oid.in_(oids))
        return pd.read_sql(query.statement, self.driver.engine)

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
                    candidate["objectId"] = oid  # Used in correction
                    candidate["step_id_corr"] = self.version
                    detections.append(candidate)

        return pd.DataFrame(detections), pd.DataFrame(non_detections)

    def cast_non_detection(self, object_id: str, candidate: dict) -> dict:
        non_detection = {
            "oid": object_id,
            "mjd": candidate["mjd"],
            "diffmaglim": candidate["diffmaglim"],
            "fid": candidate["fid"],
        }
        return non_detection

    def preprocess_lightcurves(self, detections, alerts):
        alerts.to_csv("alerts.csv")
        oids = detections.oid.values

        detections.loc[:, "parent_candid"] = None
        detections.loc[:, "has_stamp"] = True
        filtered_detections = detections.loc[:, DET_KEYS]
        filtered_detections.loc[:, "new"] = True

        light_curves = self.get_lightcurves(oids)
        light_curves["detections"]["new"] = False
        light_curves["non_detections"]["new"] = False

        # Removing already on db, similar to drop duplicates
        index_detections = pd.MultiIndex.from_frame(
            filtered_detections[["oid", "candid"]]
        )
        index_light_curve_detections = pd.MultiIndex.from_frame(
            light_curves["detections"][["oid", "candid"]]
        )
        already_on_db = index_detections.isin(index_light_curve_detections)
        filtered_detections = filtered_detections[~already_on_db]
        light_curves["detections"] = pd.concat(
            [light_curves["detections"], filtered_detections]
        )

        prv_detections = []
        prv_non_detections = []
        for _, alert in alerts.iterrows():
            if "prv_candidates" in alert:
                (
                    alert_prv_detections,
                    alert_prv_non_detections,
                ) = self.get_prv_candidates(alert)
                prv_detections.append(alert_prv_detections)
                prv_non_detections.append(alert_prv_non_detections)

        if len(prv_detections) > 0:
            prv_detections = pd.concat(prv_detections)
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
                prv_detections = self.do_correction(prv_detections)

                #   Getting columns
                current_keys = [
                    key for key in DET_KEYS if key in prv_detections.columns
                ]
                prv_detections = prv_detections.loc[:, current_keys]
                prv_detections.loc[:, "new"] = True
                light_curves["detections"] = pd.concat(
                    [light_curves["detections"], prv_detections]
                )

        if len(prv_non_detections) > 0:
            prv_non_detections = pd.concat(prv_non_detections)
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
                    [light_curves["non_detections"], prv_non_detections]
                )

        return light_curves

    def preprocess_ps1(self, metadata, detections):
        oids = metadata.oid.unique()
        metadata["new"] = False
        for i in range(1, 4):
            metadata[f"update{i}"] = False
        new_metadata = ~detections.oid.isin(oids)
        new_values = detections.loc[new_metadata, detections.columns.isin(PS1_KEYS)]
        old_values = detections.loc[~new_metadata, detections.columns.isin(PS1_KEYS)]

        if len(new_values) > 0:
            new_values.loc[:, "new"] = True
            new_values.drop_duplicates(["oid"], inplace=True)
            for i in range(1, 4):
                new_values.loc[:, f"unique{i}"] = True
                new_values.loc[:, f"update{i}"] = False
        if len(old_values) > 0:
            join_metadata = old_values.join(
                metadata.set_index("oid"), on="oid", rsuffix="_old"
            )
            for i in range(1, 4):
                difference = join_metadata[
                    np.isclose(
                        join_metadata[f"objectidps{i}"],
                        join_metadata[f"objectidps{i}_old"],
                    )
                    & join_metadata[f"unique{i}"]
                ]
                metadata[f"unique{i}"] = False
                metadata[f"update{i}"] = metadata.oid.isin(difference.oid).astype(bool)

        return pd.concat([metadata, new_values])

    def preprocess_ss(self, metadata, detections):
        oids = metadata.oid.unique()
        metadata["new"] = False
        new_metadata = ~detections.oid.isin(oids)
        new_values = detections.loc[new_metadata, detections.columns.isin(SS_KEYS)]
        if len(new_values) > 0:
            new_values.loc[:, "new"] = True

        return pd.concat([metadata, new_values])

    def preprocess_reference(self, metadata, detections):
        oids = metadata.oid.unique()
        metadata["new"] = False
        index_metadata = pd.MultiIndex.from_frame(metadata[["oid", "rfid"]])
        index_detections = pd.MultiIndex.from_frame(detections[["oid", "rfid"]])
        already_on_db = index_detections.isin(index_metadata)
        detections["mjdstartref"] = detections["jdstartref"] - 2400000.5
        detections["mjdendref"] = detections["jdendref"] - 2400000.5
        new_values = detections.loc[
            ~already_on_db, detections.columns.isin(REFERENCE_KEYS)
        ]
        if len(new_values) > 0:
            new_values.loc[:, "new"] = True

        return pd.concat([metadata, new_values])

    def preprocess_gaia(self, metadata, detections):
        tol = 1e-03
        oids = metadata.oid.unique()
        metadata[f"update1"] = False
        metadata["new"] = False
        new_metadata = ~detections.oid.isin(oids)
        new_values = detections.loc[new_metadata, detections.columns.isin(GAIA_KEYS)]
        old_values = detections.loc[~new_metadata, detections.columns.isin(GAIA_KEYS)]

        if len(new_values) > 0:
            new_values[f"unique1"] = True
            new_values[f"update1"] = False
            new_values["new"] = True

        if len(old_values) > 0:
            join_metadata = old_values.join(
                metadata.set_index("oid"), on="oid", rsuffix="_old"
            )
            is_the_same_gaia = np.isclose(
                join_metadata["maggaia"],
                join_metadata[f"maggaia_old"],
                rtol=tol,
                atol=tol,
                equal_nan=True,
            ) & np.isclose(
                join_metadata["maggaiabright"],
                join_metadata[f"maggaiabright_old"],
                rtol=tol,
                atol=tol,
                equal_nan=True,
            )
            difference = join_metadata[~(is_the_same_gaia) & join_metadata[f"unique1"]]
            metadata[f"update1"] = metadata.oid.isin(difference.oid).astype(bool)
            metadata[f"unique1"] = False
            metadata["new"] = False

        return pd.concat([metadata, new_values])

    def get_last_alert(self, alerts):
        last_alert = alerts.candid.values.argmax()
        filtered_alerts = alerts.loc[
            :, ["oid", "ndethist", "ncovhist", "jdstarthist", "jdendhist"]
        ]
        last_alert = filtered_alerts.iloc[last_alert]
        return last_alert

    def get_dataquality(self, candids):
        query = self.driver.query(Dataquality).filter(Dataquality.candid.in_(candids))
        return pd.read_sql(query.statement, self.driver.engine)

    def get_first_corrected(self, df):
        min_candid = df.candid.values.argmin()
        first_corr = df.corrected.iloc[min_candid]
        return first_corr

    def insert_detections(self, detections):
        self.logger.info(f"Inserting {len(detections)} new detections")
        detections = detections.where(pd.notnull(detections), None)
        dict_detections = detections.to_dict("records")
        self.driver.query().bulk_insert(dict_detections, Detection)

    def insert_non_detections(self, non_detections):
        self.logger.info(f"Inserting {len(non_detections)} new non_detections")
        non_detections = non_detections.where(pd.notnull(non_detections), None)
        dict_non_detections = non_detections.to_dict("records")
        self.driver.query().bulk_insert(dict_non_detections, NonDetection)

    def insert_dataquality(self, dataquality):
        # Not inserting twice
        old_dataquality = self.get_dataquality(dataquality.candid.unique().tolist())
        already_on_db = dataquality.candid.isin(old_dataquality.candid)

        dataquality = dataquality[~already_on_db]
        self.logger.info(f"Inserting {len(dataquality)} new dataquality")

        dataquality = dataquality.where(pd.notnull(dataquality), None)
        dict_dataquality = dataquality.to_dict("records")
        self.driver.query().bulk_insert(dict_dataquality, Dataquality)

    def insert_objects(self, objects):
        new_objects = objects["new"]
        objects.drop_duplicates(["oid"], inplace=True)
        objects.drop(columns=["new"], inplace=True)

        to_insert = objects[new_objects]
        to_update = objects[~new_objects]

        if len(to_insert) > 0:
            self.logger.info(f"Inserting {len(to_insert)} new objects")
            to_insert = to_insert.where(pd.notnull(to_insert), None)
            dict_to_insert = to_insert.to_dict("records")
            self.driver.query().bulk_insert(dict_to_insert, Object)

        if len(to_update) > 0:
            self.logger.info(f"Updating {len(to_update)} objects")
            to_update = to_update.where(pd.notnull(to_update), None)
            to_update.rename(columns={"oid": "_oid"}, inplace=True)
            dict_to_update = to_update.to_dict("records")
            stmt = (
                Object.__table__.update()
                .where(Object.oid == bindparam("_oid"))
                .values(OBJECT_UPDATE_PARAMS_STMT)
            )
            self.driver.engine.execute(stmt, dict_to_update)

    def insert_ss(self, metadata):
        new_metadata = metadata["new"]
        to_insert = metadata.loc[new_metadata]
        self.logger.info(f"Inserting {len(to_insert)} Solar System Metadata")
        if len(to_insert) > 0:
            to_insert = to_insert.where(pd.notnull(to_insert), None)
            dict_to_insert = to_insert.to_dict("records")
            self.driver.query().bulk_insert(dict_to_insert, Ss_ztf)

    def insert_reference(self, metadata):
        new_metadata = metadata["new"]
        to_insert = metadata[new_metadata]
        self.logger.info(f"Inserting {len(to_insert)} References")
        if len(to_insert) > 0:
            to_insert = to_insert.where(pd.notnull(to_insert), None)
            dict_to_insert = to_insert.to_dict("records")
            self.driver.query().bulk_insert(dict_to_insert, Reference)

    def insert_gaia(self, metadata):
        new_metadata = metadata["new"].astype(bool)
        to_insert = metadata[new_metadata]
        to_update = metadata[~new_metadata]

        self.logger.info(f"Inserting {len(to_insert)} Gaia Metadata")
        if len(to_insert) > 0:
            to_insert = to_insert.where(pd.notnull(to_insert), None)
            dict_to_insert = to_insert.to_dict("records")
            self.driver.query().bulk_insert(dict_to_insert, Gaia_ztf)

        self.logger.info(f"Checking {len(to_update)} Gaia unique metadata")
        if len(to_update) > 0:
            updates = to_update[to_update.update1]
            if len(updates) > 0:
                updates = updates.where(pd.notnull(updates), None)
                updates = updates[["oid", "unique1"]]
                updates.rename(columns={"oid": "_oid"}, inplace=True)
                dict_updates = updates.to_dict("records")
                self.logger.info(f"Updating {len(updates)} Gaia unique metadata")
                stmt = (
                    Gaia_ztf.__table__.update()
                    .where(Gaia_ztf.oid == bindparam("_oid"))
                    .values({"unique1": bindparam("unique1")})
                )
                self.driver.engine.execute(stmt, dict_updates)
            else:
                self.logger.info(f"No Gaia unique metadata to be updated")

    def insert_ps1(self, metadata):
        new_metadata = metadata["new"].astype(bool)
        to_insert = metadata[new_metadata]
        to_update = metadata[~new_metadata]

        self.logger.info(f"Inserting {len(to_insert)} PS1 Metadata")
        if len(to_insert) > 0:
            to_insert = to_insert.where(pd.notnull(to_insert), None)
            dict_to_insert = to_insert.to_dict("records")
            self.driver.query().bulk_insert(dict_to_insert, Ps1_ztf)

        self.logger.info(f"Checking {len(to_update)} PS1 unique metadata")
        if len(to_update) > 0:
            updates = to_update[
                to_update.update1 | to_update.update2 | to_update.update3
            ]
            if len(updates) > 0:
                updates = updates.where(pd.notnull(updates), None)
                updates = updates[["oid", "unique1", "unique2", "unique3"]]
                updates.rename(columns={"oid": "_oid"}, inplace=True)
                dict_updates = updates.to_dict("records")
                self.logger.info(f"Updating {len(updates)} PS1 unique metadata")
                stmt = (
                    Ps1_ztf.__table__.update()
                    .where(Ps1_ztf.oid == bindparam("_oid"))
                    .values(
                        {
                            "unique1": bindparam("unique1"),
                            "unique2": bindparam("unique2"),
                            "unique3": bindparam("unique3"),
                        }
                    )
                )
                self.driver.engine.execute(stmt, dict_updates)
            else:
                self.logger.info(f"No PS1 unique metadata to be updated")

    def insert_metadata(self, metadata):
        self.insert_ss(metadata["ss_ztf"])
        self.insert_reference(metadata["reference"])
        self.insert_gaia(metadata["gaia"])
        self.insert_ps1(metadata["ps1_ztf"])

    def insert_magstats(self, magstats):
        new_magstats = magstats["new"].astype(bool)
        to_insert = magstats[new_magstats]
        to_update = magstats[~new_magstats]

        self.logger.info(f"Inserting {len(to_insert)} MagStats")
        if len(to_insert) > 0:
            to_insert = to_insert.where(pd.notnull(to_insert), None)
            dict_to_insert = to_insert.to_dict("records")
            self.driver.query().bulk_insert(dict_to_insert, MagStats)

        self.logger.info(f"Updating {len(to_update)} MagStats")
        if len(to_update) > 0:
            to_update = to_update.where(pd.notnull(to_update), None)
            to_update.rename(
                columns={"oid": "_oid", "fid": "_fid", **MAGSTATS_TRANSLATE},
                inplace=True,
            )
            dict_to_update = to_update.to_dict("records")

            stmt = (
                MagStats.__table__.update()
                .where(MagStats.oid == bindparam("_oid"))
                .where(MagStats.fid == bindparam("_fid"))
                .values(MAGSTATS_UPDATE_KEYS_STMT)
            )
            self.driver.engine.execute(stmt, dict_to_update)

    def prepare_metadata(self, metadata):
        # Dropping duplicated
        for meta in ["ps1_ztf", "ss_ztf", "gaia"]:
            metadata[meta] = metadata[meta][
                ~metadata[meta].index.duplicated(keep="first")
            ]
            metadata[meta] = (
                metadata[meta].to_dict()
                if isinstance(metadata[meta], pd.Series)
                else metadata[meta].to_dict("records")[0]
            )

        for i in range(1, 4):
            metadata["ps1_ztf"][f"unique{i}"] = (
                metadata["ps1_ztf"][f"unique{i}"]
                if isinstance(metadata["ps1_ztf"][f"unique{i}"], bool)
                else bool(metadata["ps1_ztf"][f"unique{i}"])
            )
        metadata["gaia"][f"unique1"] = (
            metadata["ps1_ztf"][f"unique1"]
            if isinstance(metadata["ps1_ztf"][f"unique1"], bool)
            else bool(metadata["ps1_ztf"][f"unique1"])
        )

        data = {
            "ps1": metadata["ps1_ztf"],
            "ss": metadata["ss_ztf"],
            "gaia": metadata["gaia"],
        }
        return data

    # Skipping reference for now
    def produce(self, alerts, light_curves, metadata):
        oids = alerts.objectId.unique()
        self.logger.info(f"Checking {len(oids)} Messages")
        alerts.rename(columns={"objectId": "oid"}, inplace=True)
        alerts.set_index("oid", inplace=True)
        metadata["ps1_ztf"].set_index("oid", inplace=True)
        metadata["ss_ztf"].set_index("oid", inplace=True)
        metadata["gaia"].set_index("oid", inplace=True)
        light_curves["detections"].set_index("oid", inplace=True)
        light_curves["non_detections"].set_index("oid", inplace=True)

        n_messages = 0
        for oid in oids:
            detections = light_curves["detections"].loc[[oid]]

            max_detections = max(
                [(detections.fid == 1).sum(), (detections.fid == 2).sum()]
            )
            if max_detections < MIN_DETECTIONS_TO_PRODUCE:
                continue

            detections = detections.where(pd.notnull(detections), None)
            detections.reset_index(inplace=True)
            oid_metdata = {
                "ps1_ztf": metadata["ps1_ztf"].loc[oid],
                "ss_ztf": metadata["ss_ztf"].loc[oid],
                "gaia": metadata["gaia"].loc[oid],
            }
            oid_alerts = alerts.loc[[oid]]
            oid_candid = oid_alerts.candid.to_list()
            detections = detections.to_dict("records")

            non_detections = (
                light_curves["non_detections"].loc[[oid]]
                if oid in light_curves["non_detections"].index
                else pd.DataFrame([])
            )
            # non_detections = non_detections.where(pd.notnull(non_detections), None)
            non_detections.reset_index(inplace=True)
            non_detections = non_detections.to_dict("records")

            if "xmatches" in oid_alerts.columns:
                oid_xmatches = oid_alerts.xmatches[
                    ~oid_alerts.index.duplicated(keep="first")
                ].iloc[0]
            else:
                oid_xmatches = None

            write = {
                "oid": oid,
                "candid": oid_candid[-1],  # Change
                "detections": detections,
                "non_detections": non_detections,
                "xmatches": oid_xmatches,
                "metadata": self.prepare_metadata(oid_metdata),
                "preprocess_step_id": self.config["STEP_METADATA"]["STEP_ID"],
                "preprocess_step_version": self.config["STEP_METADATA"]["STEP_VERSION"],
            }
            self.producer.produce(write, key=oid)
            n_messages += 1
        self.logger.info(f"{n_messages} Messages Produced")

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} alerts")

        # Casting to a dataframe
        alerts = pd.DataFrame(messages)
        alerts.drop_duplicates("candid", inplace=True)
        alerts.reset_index(inplace=True)
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
        light_curves = self.preprocess_lightcurves(corrected, alerts)

        # Update dubious
        light_curves["detections"] = self.do_dubious(light_curves["detections"])

        # Getting other tables
        objects = self.get_objects(corrected["oid"].unique())
        metadata = self.get_metadata(corrected["oid"].unique())
        magstats = self.get_magstats(corrected["oid"].unique())
        metadata = self.preprocess_metadata(metadata, detections)
        new_magstats = self.do_magstats(light_curves, metadata, magstats)
        dmdt = self.do_dmdt(light_curves, new_magstats)
        new_stats = new_magstats.join(dmdt, on=["oid", "fid"])
        objects = self.preprocess_objects(objects, light_curves, detections, new_stats)

        # Insert new objects and update old objects
        self.insert_objects(objects)
        new_detections = light_curves["detections"]["new"]
        self.insert_detections(light_curves["detections"].loc[new_detections])
        self.insert_dataquality(new_dataquality)
        new_non_detections = light_curves["non_detections"]["new"]
        self.insert_non_detections(
            light_curves["non_detections"].loc[new_non_detections]
        )
        self.insert_metadata(metadata)
        self.insert_magstats(new_stats)

        alerts.to_csv("alerts.csv")

        # self.produce(alerts, light_curves, metadata)

        del alerts
        del detections
        del corrected
        del light_curves
