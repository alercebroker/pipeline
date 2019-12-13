from apf.core.step import GenericStep
import logging
import numpy as np
import pandas as pd
from apf.db.sql.models import Detection, AstroObject, NonDetection
from apf.db.sql import get_session, get_or_create
from apf.producers import KafkaProducer
import datetime
import time
import json
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
        message = self.correct_message(message)
        self.insert_db(message)
        light_curve = self.get_lightcurve(message["objectId"])
        write = {
            "oid": message["objectId"],
            "detections": light_curve["detections"],
            "non_detections": light_curve["non_detections"]
        }
        self.logger.info(write)
        self.producer.produce(write)

    def correctMagnitude(self, magref, sign, magdiff):
        result = np.nan

        try:
            aux = 10 ** (-0.4 * magref) + sign * 10 ** (-0.4 * magdiff)
            result = -2.5 * np.log10(aux)
        except:
            self.logger.exception("Correct magnitude failed")

        return result

    def correctSigmaMag(self, magref, sigmagref, sign, magdiff, sigmagdiff):

        result = np.nan

        try:
            auxref = 10 ** (-0.4 * magref)
            auxdiff = 10 ** (-0.4 * magdiff)
            aux = auxref + sign * auxdiff

            result = np.sqrt((auxref * sigmagref) ** 2 +
                             (auxdiff * sigmagdiff) ** 2) / aux

        except:

            self.logger.exception("Correct sigma magnitude failed")

        return result

    def correct_message(self, message):
        isdiffpos = str(message["candidate"]['isdiffpos'])
        isdiffpos = 1 if (isdiffpos == 't' or isdiffpos == '1') else -1
        magpsf = message["candidate"]['magpsf']
        magap = message["candidate"]['magap']
        magref = message["candidate"]['magnr']
        sigmagref = message["candidate"]['sigmagnr']
        sigmapsf = message["candidate"]['sigmapsf']
        sigmagap = message["candidate"]['sigmagap']

        magpsf_corr = self.correctMagnitude(magref, isdiffpos, magpsf)
        sigmapsf_corr = self.correctSigmaMag(
            magref, sigmagref, isdiffpos, magpsf, sigmapsf)
        magap_corr = self.correctMagnitude(magref, isdiffpos, magap)
        sigmagap_corr = self.correctSigmaMag(
            magref, sigmagref, isdiffpos, magap, sigmagap)

        message["candidate"]["magpsf_corr"] = magpsf_corr if not np.isnan(
            magpsf_corr) else None
        message["candidate"]["sigmapsf_corr"] = sigmapsf_corr if not np.isnan(
            sigmapsf_corr) else None
        message["candidate"]["magap_corr"] = magap_corr if not np.isnan(
            magap_corr) else None
        message["candidate"]["sigmagap_corr"] = sigmagap_corr if not np.isnan(
            sigmagap_corr) else None

        return message

    def insert_db(self, message):
        kwargs = {
            "mjd": message["candidate"]["jd"] - 2400000.5,
            "fid": message["candidate"]["fid"],
            "ra": message["candidate"]["ra"],
            "dec": message["candidate"]["dec"],
            "rb": message["candidate"]["rb"],
            "magap": message["candidate"]["magap"],
            "magpsf": message["candidate"]["magpsf"],
            "sigmagap": message["candidate"]["sigmagap"],
            "sigmapsf": message["candidate"]["sigmapsf"],
            "oid": message["objectId"],
            "alert": message["candidate"]
        }
        t0 = time.time()
        obj, created = get_or_create(self.session, AstroObject, filter_by={
                                     "oid": message["objectId"]})
        t1 = time.time()
        self.logger.info("object={}\tcreated={}\tdate={}\ttime={}".format(
            obj.oid, created, datetime.datetime.utcnow(), t1-t0))
        det, created = get_or_create(self.session, Detection, filter_by={
                                     "candid": str(message["candid"])}, **kwargs)
        self.logger.info("detection={}\tcreated={}\tdate={}\ttime={}".format(
            det.candid, created, datetime.datetime.utcnow(), t1-t0))

        for prv_cand in message["prv_candidates"]:
            mjd = prv_cand["jd"] - 2400000.5
            if prv_cand["diffmaglim"] is not None:
                non_detection_args = {
                    "diffmaglim": prv_cand["diffmaglim"],
                    "oid": kwargs["oid"]
                }
                non_det, created = get_or_create(self.session, NonDetection, filter_by={
                    "mjd": mjd, "fid": prv_cand["fid"]}, **non_detection_args)
                self.logger.info("non_detection={},{}\tcreated={}\tdate={}\ttime={}".format(
                    non_det.mjd, non_det.fid, created, datetime.datetime.utcnow(), t1-t0))

            else:
                detection_args = {
                    "mjd": prv_cand["jd"] - 2400000.5,
                    "fid": prv_cand["fid"],
                    "ra": prv_cand["ra"],
                    "dec": prv_cand["dec"],
                    "rb": prv_cand["rb"],
                    "magap": prv_cand["magap"],
                    "magpsf": prv_cand["magpsf"],
                    "sigmagap": prv_cand["sigmagap"],
                    "sigmapsf": prv_cand["sigmapsf"],
                    "oid": message["objectId"],
                    "alert": prv_cand
                }
                det, created = get_or_create(self.session, Detection, filter_by={
                                             "candid": str(prv_cand["candid"])}, **detection_args)
                self.logger.info("detection_in_prv_cand={}\tcreated={}\tdate={}\ttime={}".format(det.candid, created, datetime.datetime.utcnow(), t1-t0))


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
        return ret
