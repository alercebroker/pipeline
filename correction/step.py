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
    Gaia_ztf
)
from db_plugins.db.sql.serializers import (
    Gaia_ztfSchema,
    Ss_ztfSchema,
    Ps1_ztfSchema,
    ReferenceSchema
)
from db_plugins.db.sql import SQLConnection
from lc_correction.compute import apply_correction, is_dubious, apply_mag_stats, do_dmdt, DISTANCE_THRESHOLD
from astropy.time import Time
from pandas import DataFrame, Series, concat


import numpy as np
import logging
import numbers

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import Insert

@compiles(Insert)
def prefix_inserts(insert, compiler, **kw):
    return compiler.visit_insert(insert, **kw) + " ON CONFLICT DO NOTHING"

logging.getLogger("GP").setLevel(logging.WARNING)
np.seterr(divide='ignore')

DET_KEYS = ['oid', 'candid', 'mjd', 'fid', 'pid', 'diffmaglim', 'isdiffpos', 'nid', 'ra', 'dec', 'magpsf',
            'sigmapsf', 'magap', 'sigmagap', 'distnr', 'rb', 'rbversion', 'drb', 'drbversion', 'magapbig', 'sigmagapbig',
            'rfid', 'magpsf_corr', 'sigmapsf_corr', 'sigmapsf_corr_ext', 'corrected', 'dubious', 'parent_candid',
            'has_stamp', 'step_id_corr']
NON_DET_KEYS = ["oid", "mjd", "diffmaglim", "fid"]
COR_KEYS = ["magpsf_corr", "sigmapsf_corr", "sigmapsf_corr_ext"]
PS1_MultKey = [ "objectidps", "sgmag", "srmag", "simag", "szmag", "sgscore", "distpsnr"]
PS1_KEYS = ["candid","nmtchps"]
for i in range(1,4):
    PS1_KEYS = PS1_KEYS + [f"{key}{i}" for key in PS1_MultKey]
REFERENCE_KEYS = ["candid", "fid", "rcid", "field", "magnr", "sigmagnr", "chinr", "sharpnr", "chinr", "ranr", "decnr", "nframesref"]
DATAQUALITY_KEYS = ["oid","candid","fid","xpos", "ypos", "chipsf", "sky", "fwhm", "classtar", "mindtoedge", "seeratio", "aimage",
                    "bimage", "aimagerat", "bimagerat", "nneg", "nbad", "sumrat", "scorr", "magzpsci", "magzpsciunc",
                    "magzpscirms", "clrcoeff", "clrcounc", "dsnrms" , "ssnrms", "nmatches", "zpclrcov", "zpmed", "clrmed", "clrrms", "exptime"]


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
        self.version = config["STEP_VERSION"]
        self.logger.info(f"CORRECTION {self.version}")

    def get_object(self, alert: dict) -> Object:
        data = {
            "oid": alert["objectId"]
        }
        return self.driver.session.query().get_or_create(Object, filter_by=data)

    def cast_non_detection(self, object_id: str, prv_candidate: dict) -> dict:
        data = {
            "oid": object_id,
            "mjd": prv_candidate["mjd"],
            "diffmaglim": prv_candidate["diffmaglim"],
            "fid": prv_candidate["fid"],
        }
        return data

    def cast_detection(self, candidate: dict) -> dict:
        return {key: candidate[key] for key in DET_KEYS if key in candidate.keys()}

    def add_dataquality(self, candidate: dict, create=True):
        candidate_params = {}
        for key in DATAQUALITY_KEYS:
            candidate_params[key] = candidate.get(key)
        filters = {
            "candid": candidate["candid"]
        }
        data = {
            **candidate_params
        }
        if create:
            self.driver.session.query().get_or_create(Dataquality, filter_by = filters, **data)
        else:
            return {**filters,**data}

    def get_detection(self, candidate: dict) -> Detection:
        filters = {
            "candid": candidate["candid"],
            "oid": candidate["oid"]
        }
        data = self.cast_detection(candidate)
        detection, created = self.driver.session.query().get_or_create(Detection, filter_by=filters, **data)
        dataquality = self.add_dataquality(candidate)
        return detection, created


    def get_ps1(self,message: dict) -> Ps1_ztf:
        message_params = {}
        for key in PS1_KEYS:
            message_params[key] = message["candidate"][key]
        for i in range(1,4):
            unique_close = f"unique{i}"
            message_params[unique_close] = True

        filters = {
                "oid": message["objectId"],
        }
        ps1, created = self.driver.session.query().get_or_create(Ps1_ztf, filter_by=filters, **message_params)
        if not created:

            for i in range(1,4):
                objectidps = f"objectidps{i}"
                if not self.check_equal(getattr(ps1,objectidps), message_params[objectidps]):
                    unique_close = f"unique{i}"
                    setattr(ps1,unique_close, False)

        return ps1

    def get_ss(self, message: dict) -> Ss_ztf:
        params = {
            "oid": message["objectId"]
        }
        data = {
            "candid": message["candid"],
            "ssdistnr": message["candidate"]["ssdistnr"],
            "ssmagnr": message["candidate"]["ssmagnr"],
            "ssnamenr": message["candidate"]["ssnamenr"]
        }
        ss, created = self.driver.session.query().get_or_create(Ss_ztf,filter_by=params, **data)
        return ss

    def get_reference(self, message:dict) -> Reference:
        message_params = {}
        for key in REFERENCE_KEYS:
            message_params[key] = message["candidate"][key]
        data = {
            "mjdstartref": message["candidate"]["jdstartref"] - 2400000.5  ,
            "mjdendref": message["candidate"]["jdendref"] - 2400000.5,
            **message_params
        }
        filters = {
            "oid": message["objectId"],
            "rfid": message["candidate"]["rfid"]
        }
        reference, created = self.driver.session.query().get_or_create(Reference, filter_by=filters, **data)
        return reference

    def get_gaia(self,message:dict) -> Gaia_ztf:
        filters = {
            "oid": message["objectId"]
        }
        data = {
            "candid": message["candid"],
            "neargaia" : message["candidate"].get("neargaia"),
            "neargaiabright" : message["candidate"].get("neargaiabright"),
            "maggaia" : message["candidate"].get("maggaia"),
            "maggaiabright" : message["candidate"].get("maggaiabright"),
            "unique1": True
        }
        if data["neargaia"] is None:
            return
        gaia, created = self.driver.session.query().get_or_create(Gaia_ztf, filter_by=filters, **data)

        if not created:
            if not self.check_equal(gaia.neargaia, data["neargaia"]) and \
               not self.check_equal(gaia.neargaiabright, data["neargaiabright"]) and \
               not self.check_equal(gaia.maggaia, data["maggaia"]) and \
               not self.check_equal(gaia.maggaiabright, data["maggaiabright"]):
                gaia.unique1 = False

        return gaia

    def set_magstats_values(self,result: dict,magstat: MagStats) -> MagStats:
        magstat.stellar = result.stellar
        magstat.corrected = result.corrected
        magstat.ndet = int(result.ndet)
        magstat.ndubious = int(result.ndubious)
        magstat.dmdt_first = result.dmdt_first
        magstat.dm_first = result.dm_first
        magstat.sigmadm_first = result.sigmadm_first
        magstat.dt_first = result.dt_first
        magstat.magmean = result.magpsf_mean
        magstat.magmedian = result.magpsf_median
        magstat.magmax = result.magpsf_max
        magstat.magmin = result.magpsf_min
        magstat.magsigma = result.sigmapsf #I dont know this one
        magstat.maglast = result.magpsf_last
        magstat.magfirst = result.magpsf_first
        magstat.magmean_corr = result.magpsf_corr_mean
        magstat.magmedian_corr = result.magpsf_corr_median
        magstat.magmax_corr = result.magpsf_corr_max
        magstat.magmin_corr = result.magpsf_corr_min
        magstat.magsigma_corr = result.sigmapsf_corr #This doesn't exists
        magstat.maglast_corr = result.magpsf_corr_last
        magstat.magfirst_corr = result.magpsf_corr_first
        magstat.firstmjd = result.first_mjd
        magstat.lastmjd = result.last_mjd
        magstat.step_id_corr = self.version

        return MagStats

    def get_metadata(self,message: dict):
        ps1_ztf = self.get_ps1(message)
        ss_ztf = self.get_ss(message)
        reference = self.get_reference(message)
        gaia = self.get_gaia(message)
        return {
                    "ps1":ps1_ztf,
                    "ss": ss_ztf,
                    "reference": reference,
                    "gaia": gaia
                }

    def prepare_metadata(self, metadata):
        return {
            "ps1": Ps1_ztfSchema().dump(metadata["ps1"]),
            "ss": Ss_ztfSchema().dump(metadata["ss"]),
            "reference": ReferenceSchema().dump(metadata["reference"]),
            "gaia": Gaia_ztfSchema().dump(metadata["gaia"])
        }

    def get_magstats(self, message: dict, light_curve: list, ss: Ss_ztf, ps1: Ps1_ztf, reference: Reference) -> MagStats:

        detections = DataFrame(light_curve["detections"])
        non_detections = DataFrame(light_curve["non_detections"])
        detections.index = detections.candid
        detections_fid = detections[detections.fid == message["candidate"]["fid"]]


        new_stats = apply_mag_stats(detections_fid,
                                    distnr=ss.ssdistnr,
                                    distpsnr1=ps1.distpsnr1,
                                    sgscore1=ps1.sgscore1,
                                    chinr=reference.chinr,
                                    sharpnr=reference.sharpnr)

        if len(non_detections) > 0:
            non_detections_fid = non_detections[non_detections.fid == message["candidate"]["fid"]]
            new_stats_dmdt = do_dmdt(non_detections_fid,new_stats)
        else:
            new_stats_dmdt = Series({
                'dmdt_first': np.nan,
                'dm_first': np.nan,
                'sigmadm_first': np.nan,
                'dt_first': np.nan
            })
        all_stats = concat([new_stats,new_stats_dmdt])
        filters = {
            "oid": message["objectId"],
            "fid": message["candidate"]["fid"]
        }
        magStats, created = self.driver.session.query().get_or_create(MagStats,filter_by=filters)
        self.set_magstats_values(all_stats, magStats)

    def set_object_values(self, alert: dict, obj: Object) -> Object:
        obj.ndethist = alert["candidate"]["ndethist"]
        obj.ncovhist = alert["candidate"]["ncovhist"]
        obj.mjdstarthist = alert["candidate"]["jdstarthist"] - 2400000.5
        obj.mjdendhist = alert["candidate"]["jdendhist"] - 2400000.5
        obj.firstmjd = alert["candidate"]["jd"] - 2400000.5
        obj.lastmjd = obj.firstmjd
        obj.ndet = 1
        obj.deltamjd = 0
        obj.meanra = alert["candidate"]["ra"]
        obj.meandec = alert["candidate"]["dec"]
        obj.step_id_corr = self.version
        return obj

    def preprocess_alert(self, alert: dict, is_prv_candidate=False) -> None:
        if is_prv_candidate:
            alert["mjd"] = alert["jd"] - 2400000.5
            alert["isdiffpos"] = 1 if alert["isdiffpos"] in ["t", "1"] else -1
            alert["corrected"] = alert["distnr"] < DISTANCE_THRESHOLD
            alert["candid"] = alert["candid"]
            alert["has_stamp"] = False
            alert["step_id_corr"] = self.version
        else:
            alert["candidate"]["mjd"] = alert["candidate"]["jd"] - 2400000.5
            alert["candidate"]["isdiffpos"] = 1 if alert["candidate"]["isdiffpos"] in ["t", "1"] else -1
            alert["candidate"]["corrected"] = alert["candidate"]["distnr"] < DISTANCE_THRESHOLD
            alert["candidate"]["candid"] = alert["candidate"]["candid"]
            alert["candidate"]["has_stamp"] = True
            alert["candidate"]["step_id_corr"] = self.version
            for k in ["cutoutDifference", "cutoutScience", "cutoutTemplate"]:
                alert.pop(k, None)

    def do_correction(self, candidate: dict, obj: Object, inplace=False) -> dict:
        values = apply_correction(candidate)
        result = dict(zip(COR_KEYS, values))
        corr_magstats = candidate["corrected"]
        first_mjd = obj.firstmjd
        for det in obj.detections:
            if first_mjd == det.mjd:
                corr_magstats = det["corrected"]
        candidate["dubious"] = bool(is_dubious(candidate["corrected"], candidate["isdiffpos"], corr_magstats))
        if inplace:
            candidate.update(result)
        return result

    def check_equal(self,value_1,value_2):
        if isinstance(value_1, numbers.Number):
            return abs(value_2 - value_1) < 1e-5
        else:
            return value_1 == value_2


    def already_exists(self, candidate: dict, candidates_list: list, keys: list) -> bool:
        for cand in candidates_list:
            if all([self.check_equal(candidate[k],cand[k]) for k in keys]):
                return True
        return False

    def check_candid_in_db(self, oid, candid):
        query = self.driver.session.query(Detection.oid, Detection.candid).filter_by(oid=oid,candid=candid)
        result = query.scalar()
        exists = result is not None
        return exists

    def process_prv_candidates(self, prv_candidates: dict, obj: Object, parent_candid: str, light_curve: dict) -> None:
        prv_non_detections = []
        prv_detections = []
        prv_dataquality = []
        if prv_candidates is None:
            return
        for prv in prv_candidates:
            is_non_detection = prv["candid"] is None
            prv["mjd"] = prv["jd"] - 2400000.5

            if is_non_detection:
                non_detection = self.cast_non_detection(obj.oid, prv)
                if not self.already_exists(non_detection, light_curve["non_detections"], ["mjd"]):
                    light_curve["non_detections"].append(non_detection)
                    prv_non_detections.append(non_detection)
            else:
                self.preprocess_alert(prv, is_prv_candidate=True)
                if not self.already_exists(prv, light_curve["detections"], ["candid"]) and not self.check_candid_in_db(obj.oid,prv["candid"]):
                    self.do_correction(prv, obj, inplace=True)
                    dataquality = self.add_dataquality(prv,create=False)
                    dataquality["oid"] = obj.oid
                    prv["oid"] = obj.oid
                    prv["parent_candid"] = parent_candid
                    light_curve["detections"].append(prv)
                    prv_detections.append(prv)
                    prv_dataquality.append(dataquality)

        # Insert data to database
        self.driver.session.query().bulk_insert(prv_detections, Detection)
        self.driver.session.query().bulk_insert(prv_dataquality, Dataquality)
        self.driver.session.query().bulk_insert(prv_non_detections, NonDetection)

    def process_lightcurve(self, alert: dict, obj: Object) -> dict:
        # Setting identifier of object to detection
        alert["candidate"]["oid"] = obj.oid
        detection, created = self.get_detection(alert["candidate"])
        light_curve = self.get_light_curve(obj)
        if created:
            detection = detection.__dict__
            del detection["_sa_instance_state"]
            light_curve["detections"].append(detection)

        else:
            self.logger.warning(f"[{obj.oid}-{detection.candid}] Detection already exists")
        # Compute and analyze previous candidates
        if "prv_candidates" in alert:
            self.process_prv_candidates(alert["prv_candidates"], obj, detection["candid"], light_curve)

        return light_curve

    def get_light_curve(self, obj: Object) -> dict:
        detections = obj.detections
        non_detections = obj.non_detections
        return {
            "detections": list(map(lambda x: {k: x[k] for k in DET_KEYS}, detections)),
            "non_detections": list(map(lambda x: {k: x[k] for k in NON_DET_KEYS}, non_detections)),
        }

    def set_basic_stats(self, detections: list, obj: Object) -> None:
        detections = DataFrame.from_dict(detections)
        obj.ndet = len(detections)
        obj.lastmjd = detections["mjd"].max()
        obj.firstmjd = detections["mjd"].min()
        obj.meanra = detections["ra"].mean()
        obj.meandec = detections["dec"].mean()
        obj.sigmara = detections["ra"].std()
        obj.sigmadec = detections["dec"].mean()
        obj.deltamjd = obj.lastmjd - obj.firstmjd

    def execute(self, message):
        self.logger.info(f'[{message["objectId"]}-{message["candid"]}] Processing message')
        obj, created = self.get_object(message)
        self.preprocess_alert(message)
        self.do_correction(message["candidate"], obj, inplace=True)
        light_curve = self.process_lightcurve(message, obj)
        metadata = self.get_metadata(message)
        magstats = self.get_magstats(message,
                                     light_curve,
                                     ps1 = metadata["ps1"],
                                     ss = metadata["ss"],
                                     reference=metadata["reference"])

        # First observation of the object
        if created:
            self.set_object_values(message, obj)

        # When the object + prv_candidates > 1
        if len(light_curve["detections"]) > 1:
            self.set_basic_stats(light_curve["detections"], obj)

        # Write in database
        self.driver.session.commit()
        self.logger.info(f'[{message["objectId"]}-{message["candid"]}] Messages processed')
        write = {
            "oid": message["objectId"],
            "candid": message["candid"],
            "detections": light_curve["detections"],
            "non_detections": light_curve["non_detections"],
            "xmatches": message.get("xmatches"),
            "fid": message["candidate"]["fid"],
            "metadata": self.prepare_metadata(metadata)
        }
        self.producer.produce(write)
