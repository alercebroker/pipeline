from apf.core import get_class
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from db_plugins.db.sql.models import Object, Detection, NonDetection
from db_plugins.db.sql import SQLConnection
from lc_correction.compute import apply_correction, is_dubious, DISTANCE_THRESHOLD
from astropy.time import Time
from pandas import DataFrame


import numpy as np
import logging

logging.getLogger("GP").setLevel(logging.WARNING)
np.seterr(divide='ignore')

DET_KEYS = ['avro', 'oid', 'candid', 'mjd', 'fid', 'pid', 'diffmaglim', 'isdiffpos', 'nid', 'ra', 'dec', 'magpsf',
            'sigmapsf', 'magap', 'sigmagap', 'distnr', 'rb', 'rbversion', 'drb', 'drbversion', 'magapbig', 'sigmagapbig',
            'rfid', 'magpsf_corr', 'sigmapsf_corr', 'sigmapsf_corr_ext', 'corrected', 'dubious', 'parent_candid',
            'has_stamp', 'step_id_corr']
NON_DET_KEYS = ["oid", "mjd", "diffmaglim", "fid", "datetime"]
COR_KEYS = ["magpsf_corr", "sigmapsf_corr", "sigmapsf_corr_ext"]


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
        return self.driver.session.query().get_or_create(Object, data)

    def cast_non_detection(self, object_id: str, prv_candidate: dict) -> dict:
        data = {
            "oid": object_id,
            "mjd": prv_candidate["mjd"],
            "diffmaglim": prv_candidate["diffmaglim"],
            "fid": prv_candidate["fid"],
            "datetime": Time(prv_candidate["mjd"], format="mjd").datetime
        }
        return data

    def cast_detection(self, candidate: dict) -> dict:
        return {key: candidate[key] for key in DET_KEYS if key in candidate.keys()}

    def get_detection(self, candidate: dict) -> Detection:
        filters = {
            "candid": candidate["candid"]
        }
        data = self.cast_detection(candidate)
        return self.driver.session.query().get_or_create(Detection, filter_by=filters, **data)

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
            alert["candid"] = str(alert["candid"])
            alert["has_stamp"] = False
            alert["step_id_corr"] = self.version
        else:
            alert["candidate"]["mjd"] = alert["candidate"]["jd"] - 2400000.5
            alert["candidate"]["isdiffpos"] = 1 if alert["candidate"]["isdiffpos"] in ["t", "1"] else -1
            alert["candidate"]["corrected"] = alert["candidate"]["distnr"] < DISTANCE_THRESHOLD
            alert["candidate"]["candid"] = str(alert["candidate"]["candid"])
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

    def already_exists(self, candidate: dict, candidates_list: list, keys: list) -> bool:
        for cand in candidates_list:
            if all([candidate[k] == cand[k] for k in keys]):
                return True
        return False

    def process_prv_candidates(self, prv_candidates: dict, obj: Object, parent_candid: str, light_curve: dict) -> None:
        prv_non_detections = []
        prv_detections = []
        for prv in prv_candidates:
            is_non_detection = prv["candid"] is None
            prv["mjd"] = prv["jd"] - 2400000.5

            if is_non_detection:
                non_detection = self.cast_non_detection(obj.oid, prv)
                if not self.already_exists(non_detection, light_curve["non_detections"], ["datetime", "fid", "oid"]):
                    light_curve["non_detections"].append(non_detection)
                    prv_non_detections.append(non_detection)
            else:
                self.preprocess_alert(prv, is_prv_candidate=True)
                if not self.already_exists(prv, light_curve["detections"], ["candid"]):
                    self.do_correction(prv, obj, inplace=True)
                    prv["oid"] = obj.oid
                    prv["parent_candid"] = parent_candid
                    light_curve["detections"].append(prv)
                    prv_detections.append(prv)
        # Insert data to database
        self.driver.session.query().bulk_insert(prv_detections, Detection)
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
            self.logger.warning(f"{detection} already exists")
        # Compute and analyze previous candidates
        if "prv_candidates" in alert:
            self.process_prv_candidates(alert["prv_candidates"], obj, detection["candid"], light_curve)
        # Remove datetime key in light_curve. Reason: Not necessary when this values passed to other steps
        for non_det in light_curve["non_detections"]:
            if "datetime" in non_det:
                del non_det["datetime"]
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
        obj, created = self.get_object(message)
        self.logger.info(obj)
        self.preprocess_alert(message)
        self.do_correction(message["candidate"], obj, inplace=True)
        light_curve = self.process_lightcurve(message, obj)
        # First observation of the object
        if created:
            self.set_object_values(message, obj)
        # When the object + prv_candidates > 1
        if len(light_curve["detections"]) > 1:
            self.logger.warning("Do basic stats")
            self.set_basic_stats(light_curve["detections"], obj)

        # Write in database
        self.driver.session.commit()
        write = {
            "oid": message["objectId"],
            "candid": str(message["candid"]),
            "detections": light_curve["detections"],
            "non_detections": light_curve["non_detections"],
            "fid": message["candidate"]["fid"]
        }
        self.producer.produce(write)
