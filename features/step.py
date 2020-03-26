from apf.core.step import GenericStep
from apf.producers import KafkaProducer
import logging

from late_classifier.features.preprocess import *
from late_classifier.features.custom import *

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

from apf.db.sql import get_session, get_or_create
from apf.db.sql.models import Features, AstroObject, FeaturesObject


import datetime
import time

import warnings
warnings.filterwarnings('ignore')
logging.getLogger("GP").setLevel(logging.WARNING)


class FeaturesComputer(GenericStep):
    """FeaturesComputer Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """
    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)
        self.preprocessor = DetectionsPreprocessorZTF()
        self.featuresComputer = CustomHierarchicalExtractor()#HierarchicalExtractor([1, 2])
        self.session = get_session(self.config["DB_CONFIG"])

        prod_config = self.config.get("PRODUCER_CONFIG", None)
        if prod_config:
            self.producer = KafkaProducer(prod_config)
        else:
            self.producer = None

    def _clean_result(self, result):
        cleaned_results = {}
        for key in result:
            if type(result[key]) is dict:
                cleaned_results[key] = self._clean_result(result[key])
            else:
                if np.isnan(result[key]):
                    cleaned_results[key] = None
                else:
                    cleaned_results[key] = result[key]
        return cleaned_results

    def execute(self, message):
        t0 = time.time()
        oid = message["oid"]
        detections = json_normalize(message["detections"])
        non_detections = json_normalize(message["non_detections"])
        if len(detections) < 6:
            self.logger.debug(f"{oid} Object has less than 6 detections")
            return
        else:
            self.logger.debug(f"{oid} Object has enough detections")
            self.logger.debug("Calculating Features")
        detections.rename(columns={
            "alert.sgscore1": "sgscore1",
            "alert.isdiffpos": "isdiffpos",
        }, inplace=True)
        detections.index = [oid]*len(detections)
        features_t0 = time.time()
        features = self.featuresComputer.compute_features(detections, non_detections=non_detections)
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_t1 = time.time()

        if len(features) > 0:
            if type(features) is pd.Series:
                features = pd.DataFrame([features])
            result = self._clean_result(features.loc[oid].to_dict())
        else:
            self.logger.debug(f"No features for {oid}")
            return

        obj, created = get_or_create(self.session, AstroObject, filter_by={"oid": oid})
        version, created = get_or_create(self.session, Features, filter_by={"version": self.config["FEATURE_VERSION"]})
        features, created = get_or_create(self.session, FeaturesObject, filter_by={
            "features_version": self.config["FEATURE_VERSION"], "object_id": oid
        })

        features.data = result
        features.features = version
        obj.features.append(features)
        self.session.commit()

        if self.producer:
            out_message = {
                "oid": oid,
                "features": result
            }
            self.producer.produce(out_message)

        compute_time = features_t1-features_t0
        wall_time = time.time()-t0

        self.send_metrics(oid=oid, compute_time=compute_time)
        self.logger.info(f"object={oid}\tdate={datetime.datetime.now()}\tcompute_time={compute_time:.6f}\twall_time={wall_time:.6f}")
