from apf.core.step import GenericStep
import logging
import numpy as np
import pandas as pd
from apf.db.sql.models import Detection, AstroObject, NonDetection
from apf.db.sql import get_session, get_or_create, check_exists,bulk_insert
from apf.producers import KafkaProducer
from apf.consumers import KafkaConsumer
import datetime
import math
import time
import json
import io
from astropy.time import Time
np.seterr(divide='ignore')


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
        self.session = get_session(config["DB_CONFIG"])
        self.producer = KafkaProducer(config["PRODUCER_CONFIG"])

    def execute(self, message):
        message["candidate"].update(self.correct_message(message["candidate"]))
        light_curve = self.get_lightcurve(message["objectId"])
        self.insert_db(message, light_curve)
        write = {
            "oid": message["objectId"],
            "detections": light_curve["detections"],
            "non_detections": light_curve["non_detections"]
        }
        self.producer.produce(write)

    def correctMagnitude(self, magref, sign, magdiff):
        result = np.nan

        try:
            aux = np.power(10, (-0.4 * magref)) + sign * np.power(10,(-0.4 * magdiff))
            result = -2.5 * np.log10(aux)
        except:
            self.logger.exception("Correct magnitude failed")

        return result

    def correctSigmaMag(self, magref, sigmagref, sign, magdiff, sigmagdiff):

        result = np.nan

        try:
            auxref = np.power(10, (-0.4 * magref))
            auxdiff = np.power(10,(-0.4 * magdiff))
            aux = auxref + sign * auxdiff

            result = np.sqrt(np.power((auxref * sigmagref), 2) +
                             np.power((auxdiff * sigmagdiff), 2)) / aux

        except:

            self.logger.exception("Correct sigma magnitude failed")

        return result

    def correct_message(self, message):
        isdiffpos = str(message['isdiffpos'])
        isdiffpos = 1 if (isdiffpos == 't' or isdiffpos == '1') else -1
        message["isdiffpos"] = isdiffpos
        magpsf = message['magpsf']
        magap = message['magap']
        magref = message['magnr']
        sigmagref = message['sigmagnr']
        sigmapsf = message['sigmapsf']
        sigmagap = message['sigmagap']

        magpsf_corr = self.correctMagnitude(magref, isdiffpos, magpsf)
        sigmapsf_corr = self.correctSigmaMag(
            magref, sigmagref, isdiffpos, magpsf, sigmapsf)
        magap_corr = self.correctMagnitude(magref, isdiffpos, magap)
        sigmagap_corr = self.correctSigmaMag(
            magref, sigmagref, isdiffpos, magap, sigmagap)

        message["magpsf_corr"] = magpsf_corr if not np.isnan(
            magpsf_corr) and not np.isinf(magpsf_corr) else None
        message["sigmapsf_corr"] = sigmapsf_corr if not np.isnan(
            sigmapsf_corr) and not np.isinf(sigmapsf_corr) else None
        message["magap_corr"] = magap_corr if not np.isnan(
            magap_corr) and not np.isinf(magap_corr) else None
        message["sigmagap_corr"] = sigmagap_corr if not np.isnan(
            sigmagap_corr) and not np.isinf(sigmagap_corr) else None

        return message

    def insert_db(self, message, light_curve):
        kwargs = {
            "mjd": message["candidate"]["jd"] - 2400000.5,
            "fid": message["candidate"]["fid"],
            "ra": message["candidate"]["ra"],
            "dec": message["candidate"]["dec"],
            "rb": message["candidate"]["rb"],
            "magap": message["candidate"]["magap"],
            "magap_corr": message["candidate"]["magap_corr"],
            "magpsf": message["candidate"]["magpsf"],
            "magpsf_corr": message["candidate"]["magpsf_corr"],
            "sigmagap": message["candidate"]["sigmagap"],
            "sigmagap_corr": message["candidate"]["sigmagap_corr"],
            "sigmapsf": message["candidate"]["sigmapsf"],
            "sigmapsf_corr": message["candidate"]["sigmapsf_corr"],
            "oid": message["objectId"],
            "alert": message["candidate"],
        }
        found = list(filter(lambda det: det['candid'] == str(message["candid"]), light_curve["detections"]))
        self.logger.info(len(found))
        if len(found) > 0:
            return

        t0 = time.time()
        obj, created = get_or_create(self.session, AstroObject, filter_by={
            "oid": message["objectId"]})
        t1 = time.time()
        self.logger.debug("object={}\tcreated={}\tdate={}\ttime={}".format(
            obj.oid, created, datetime.datetime.utcnow(), t1-t0))
        t0 = time.time()
        det, created = get_or_create(self.session, Detection, filter_by={
            "candid": str(message["candid"])}, **kwargs)
        t1 = time.time()
        self.logger.debug("detection={}\tcreated={}\tdate={}\ttime={}".format(
            det.candid, created, datetime.datetime.utcnow(), t1-t0))

        prv_cands = []
        non_dets = []
        if message["prv_candidates"]:
            t0 = time.time()
            for prv_cand in message["prv_candidates"]:
                mjd = prv_cand["jd"] - 2400000.5
                if prv_cand["diffmaglim"] is not None:
                    non_detection_args = {
                        "diffmaglim": prv_cand["diffmaglim"],
                        "oid": kwargs["oid"],
                        "mjd": mjd
                    }

                    dt = Time(mjd,format="mjd")
                    filters = {"datetime":  dt.datetime, "fid": prv_cand["fid"], "oid": message["objectId"]}
                    found = list(filter(lambda non_det: (Time(non_det["mjd"], format="mjd").datetime == filters["datetime"]) and
                                                        (non_det["fid"] == filters["fid"]) and
                                                        (non_det["oid"] == filters["oid"]), light_curve["non_detections"]))
                    if len(found) == 0:
                        non_detection_args.update(filters)
                        if non_detection_args not in non_dets:
                            non_dets.append(non_detection_args)
                else:
                    found = list(filter(lambda det: det['candid'] == prv_cand["candid"], light_curve["detections"]))
                    if len(found) == 0:
                        prv_cand.update(self.correct_message(prv_cand))
                        detection_args = {
                            "mjd": prv_cand["jd"] - 2400000.5,
                            "fid": prv_cand["fid"],
                            "ra": prv_cand["ra"],
                            "dec": prv_cand["dec"],
                            "rb": prv_cand["rb"],
                            "magap": prv_cand["magap"],
                            "magap_corr": prv_cand["magap_corr"],
                            "magpsf": prv_cand["magpsf"],
                            "magpsf_corr": prv_cand["magpsf_corr"],
                            "sigmagap": prv_cand["sigmagap"],
                            "sigmagap_corr": prv_cand["sigmagap_corr"],
                            "sigmapsf": prv_cand["sigmapsf"],
                            "sigmapsf_corr": prv_cand["sigmapsf_corr"],
                            "oid": message["objectId"],
                            "alert": prv_cand,
                            "candid": str(prv_cand["candid"]),
                            "parent_candidate": str(message["candid"])
                        }
                        prv_cands.append(detection_args)

            bulk_insert(prv_cands, Detection, self.session)
            bulk_insert(non_dets, NonDetection, self.session)
            t1 = time.time()
            self.logger.debug("Processed {} prv_candidates in {} seconds".format(
                len(message["prv_candidates"]), t1-t0))
        self.session.commit()

    def get_lightcurve(self, oid):
        detections = self.session.query(Detection).filter_by(oid=oid)
        non_detections = self.session.query(NonDetection).filter_by(oid=oid)
        ret = {
            "detections": [d.__dict__ for d in detections.all()],
            "non_detections": [d.__dict__ for d in non_detections.all()]
        }
        for d in ret["detections"]:
            del d["_sa_instance_state"]
        for d in ret["non_detections"]:
            del d['_sa_instance_state']
            del d['datetime']
        return ret

    def jd_to_date(self, jd):
        jd = jd + 0.5
        F, I = math.modf(jd)
        I = int(I)
        A = math.trunc((I - 1867216.25)/36524.25)
        if I > 2299160:
            B = I + 1 + A - math.trunc(A / 4.)
        else:
            B = I
        C = B + 1524
        D = math.trunc((C - 122.1) / 365.25)
        E = math.trunc(365.25 * D)
        G = math.trunc((C - E) / 30.6001)
        day = C - E + F - math.trunc(30.6001 * G)
        if G < 13.5:
            month = G - 1
        else:
            month = G - 13
        if month > 2.5:
            year = D - 4716
        else:
            year = D - 4715
        return year, month, day
