from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from apf.db.sql import get_or_create
from apf.db.sql.models import Class, Classification, Classifier

import logging

import os
import pandas as pd
import numpy as np
class LateClassifier(GenericStep):
    """LateClassifier Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """
    def __init__(self,consumer = None, config = None,level = logging.INFO,**step_args):
        super().__init__(consumer,config=config, level=level)

        BASE_PATH = os.path.dirname(os.path.realpath(__file__))
        self.logger.info("Loading Models")
        root_path = os.path.join(BASE_PATH, "models/root.pkl")
        root_dict = pd.read_pickle(root_path)

        periodic_path = os.path.join(BASE_PATH, "models/periodic.pkl")
        periodic_dict = pd.read_pickle(periodic_path)

        stochastic_path = os.path.join(BASE_PATH, "models/stochastic.pkl")
        stochastic_dict = pd.read_pickle(stochastic_path)

        transient_path = os.path.join(BASE_PATH, "models/transient.pkl")
        transient_dict = pd.read_pickle(transient_path)

        self.root = root_dict['rf_model']
        self.root_classes = root_dict['order_classes']
        self.childs = {
            "Stochastic":{
                "model":stochastic_dict["rf_model"],
                "classes":stochastic_dict["order_classes"]
            },
            "Periodic":{
                "model":periodic_dict["rf_model"],
                "classes":periodic_dict["order_classes"]
            },
            "Transient":{
                "model":transient_dict["rf_model"],
                "classes":transient_dict["order_classes"]
            },

        }
        self.feature_list = root_dict['features']
        if self.config.get("PRODUCER_CONFIG", None):
            self.producer = KafkaProducer(self.config["PRODUCER_CONFIG"])
        else:
            self.producer = None

    def execute(self,message):
        oid = message["oid"]

        ## HACK:
        message["features"]["n_samples_1"] = message["features"].get("n_det_1", np.nan)
        message["features"]["n_samples_2"] = message["features"].get("n_det_2", np.nan)

        #Preprocess
        features = pd.Series(message["features"])
        features.replace([np.inf, -np.inf], np.nan,inplace=True)
        features.fillna(-999, inplace=True)

        #Checking if all exists
        feature_exists = [feature in features.index for feature in self.feature_list]
        has_all_features = all(feature_exists)
        if not has_all_features:
            not_found = [feature for feature in self.feature_list if feature not in features.index]
            self.logger.info(f"{oid}\t Missing Features: {not_found}")
        else:
            features = features[self.feature_list].values.reshape((1,-1))

            probs_root = dict(zip(self.root_classes,self.root.predict_proba(features)[0]))

            probs_childs = {}
            for key in self.childs:
                probs_child = dict(zip(self.childs[key]["classes"],self.childs[key]["model"].predict_proba(features)[0]))
                probs_childs[key] = probs_child

            probs_all = {}
            for key in probs_root:
                base = probs_root[key]
                for ckey in probs_childs[key]:
                    probs_all[ckey] = probs_childs[key][ckey] * base

            max_prob = 0
            for key in probs_all:
                if max_prob < probs_all[key]:
                    selected_class = key
                    max_prob = probs_all[key]

            result = {
                "oid": oid,
                "hierarchical": [probs_root,probs_childs],
                "probabilties": probs_all,
                "class": selected_class
                }

            self.producer.produce(result)
            self.logger.info(result)
            raise
