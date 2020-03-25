from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from late_classifier.classifier.hierarchical_rf import HierarchicalRF

import logging
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
    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)

        self.logger.info("Loading Models")
        self.model = HierarchicalRF()
        self.features_required = set(self.model.model["features"])

        if self.config.get("PRODUCER_CONFIG", None):
            self.producer = KafkaProducer(self.config["PRODUCER_CONFIG"])
        else:
            self.producer = None

    def execute(self, message):
        oid = message["oid"]

        message["features"]["n_samples_1"] = message["features"].get("n_det_1", np.nan)
        message["features"]["n_samples_2"] = message["features"].get("n_det_2", np.nan)

        features = pd.Series(message["features"])

        missing_features = self.features_required.difference(set(features.index))

        if len(missing_features) != 0:
            self.logger.info(f"{oid}\t Missing Features: {missing_features}")
        else:
            features.replace([np.inf, -np.inf], np.nan, inplace=True)
            features.fillna(-999, inplace=True)
            features = features[self.features_required].values.reshape((1, -1))
            result = self.model.predict(features, pipeline=True)
            result["oid"] = oid
            self.producer.produce(result)
            self.logger.info(result)
