from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from late_classifier.classifier.models import HierarchicalRandomForest, PICKLE_PATH

import logging
import pandas as pd
import numpy as np

from apf.db.sql import get_or_create, get_session, update
from apf.db.sql.models import Classification, Classifier, Class, AstroObject, Taxonomy


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
        self.model = HierarchicalRandomForest({})
        self.model.load_model(PICKLE_PATH)
        self.features_required = set(self.model.feature_list)

        prod_config = self.config.get("PRODUCER_CONFIG", None)
        if prod_config:
            self.producer = KafkaProducer(prod_config)
        else:
            self.producer = None

        self.session = get_session(config["DB_CONFIG"])

    def _format_features(self, features):
        features = pd.DataFrame.from_records([features])
        features.rename(columns={
            "PeriodLS_v2_1": 'MHAOV_Period_1',
            "PeriodLS_v2_2": 'MHAOV_Period_2',
            "Period_fit_v2_1": 'Period_fit_1',
            "Period_fit_v2_2": 'Period_fit_2',
            "Psi_CS_v2_1": 'Psi_CS_1',
            "Psi_CS_v2_2": 'Psi_CS_2',
            'Psi_eta_v2_1': 'Psi_eta_1',
            'Psi_eta_v2_2': 'Psi_eta_2'
        }, inplace=True)
        return features

    def insert_db(self, result, oid):
        classifier, _ = get_or_create(self.session, Classifier, filter_by={
            "name": self.config["CLASSIFIER_NAME"]})
        taxonomy, _ = get_or_create(self.session, Taxonomy, filter_by={
            "name": self.config["TAXONOMY_NAME"]})
        taxonomy.classifiers.append(classifier)
        _class, _ = get_or_create(self.session, Class, filter_by={
            "name": result["class"]})
        kwargs = {
            "probability": result["probabilities"][max(result["probabilities"], key=result["probabilities"].get)],
            "probabilities": result["probabilities"],
            "class_name": _class.name
        }

        resp, created = get_or_create(self.session, Classification, filter_by={
            "astro_object": oid, "classifier_name": classifier.name}, **kwargs)
        if not created:
            resp = update(resp, kwargs)
        self.session.commit()

    def execute(self, message):
        oid = message["oid"]
        message["features"]["n_samples_1"] = message["features"].get(
            "n_det_1", np.nan)
        message["features"]["n_samples_2"] = message["features"].get(
            "n_det_2", np.nan)

        features = self._format_features(message["features"])
        missing_features = self.features_required.difference(set(features.columns))

        if len(missing_features) != 0:
            self.logger.debug(f"{oid}\t Missing Features: {missing_features}")
        else:
            features.replace([np.inf, -np.inf], np.nan, inplace=True)
            features.fillna(-999, inplace=True)
            features = features[self.features_required]
            result = self.model.predict_in_pipeline(features)

            self.insert_db(result, oid)

            new_message = {
                "oid": oid,
                "features": message["features"],
                "late_classification": result,
            }
            self.logger.debug(new_message)
            self.producer.produce(new_message)
