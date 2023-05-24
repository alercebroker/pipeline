from apf.core import get_class
from apf.core.step import GenericStep
from lc_classifier.classifier.models import HierarchicalRandomForest
from lc_classification.predictors.utils.no_class_post_processor import (
    NoClassifiedPostProcessor,
)

import logging
import pandas as pd
import numpy as np
import json

import numexpr

from lc_classification.predictors.ztf_random_forest.ztf_random_forest_parser import (
    ZtfRandomForestPredictorParser,
)


class LateClassifier(GenericStep):
    """Light Curve Classification Step, for a description of the algorithm used to process
    check the `execute()` method.

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    base_name = "lc_classifier"

    def __init__(self, config=None, level=logging.INFO, model=None, **step_args):
        super().__init__(config=config, level=level)
        numexpr.utils.set_num_threads(1)
        self.logger.info("Loading Models")
        self.model = model or HierarchicalRandomForest({})
        self.model.download_model()
        self.model.load_model(self.model.MODEL_PICKLE_PATH)
        cls = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(config["SCRIBE_PRODUCER_CONFIG"])
        self.predictor_parser = ZtfRandomForestPredictorParser(self.model.feature_list)

    def get_aid_tree(self, tree, aid):
        tree_aid = {}
        for key in tree:
            data = tree[key]
            if isinstance(data, pd.DataFrame):
                tree_aid[key] = data.loc[aid].to_dict()
            elif isinstance(data, pd.Series):
                tree_aid[key] = data.loc[aid]
            elif isinstance(data, dict):
                tree_aid[key] = self.get_aid_tree(data, aid)
        return tree_aid

    def pre_produce(self, result: tuple):
        alert_data, features, tree_probabilities = result
        messages = []
        self.metrics["class"] = tree_probabilities["class"].tolist()
        features.drop(columns=["candid"], inplace=True)
        features.replace({np.nan: None}, inplace=True)
        alert_data.sort_values("candid", ascending=False, inplace=True)
        alert_data.drop_duplicates("aid", inplace=True)
        for _, row in alert_data.iterrows():
            aid = row.aid
            candid = row.candid
            features_aid = features.loc[aid].to_dict()

            tree_aid = self.get_aid_tree(tree_probabilities, aid)
            write = {
                "aid": aid,
                "candid": candid,
                "features": features_aid,
                "lc_classification": tree_aid,
            }
            messages.append(write)

        return messages

    def message_to_df(self, messages):
        return pd.DataFrame(
            [
                {"aid": message.get("aid"), "candid": message.get("candid", np.nan)}
                for message in messages
            ]
        )

    def features_to_df(self, alert_data, messages):
        features = pd.DataFrame([message["features"] for message in messages])
        features["aid"] = alert_data.aid
        features["candid"] = alert_data.candid
        features.sort_values("candid", ascending=False, inplace=True)
        features.drop_duplicates("aid", inplace=True)
        return features

    def produce_scribe(self, classifications: list):
        for classification in classifications:
            aid = classification.aid
            command = {
                "collection": "object",
                "type": "update_probabilities",
                "criteria": {"_id": aid},
                "data": classification.classification,
                "options": {"upsert": True, "set_on_insert": False},
            }
            self.scribe_producer.produce({"payload": json.dumps(command)})

    def execute(self, messages):
        """Run the classification.
        1.- Get the features and transform them to a pd.DataFrame.
        2.- Check if there are missing features.
        3.- Do inference on the object features to get the probabilities.
        4.- Insert into the database.
            4.1.- Get the ranking for each level.
            4.2.- Store the Taxonomy if necessary.
            4.3.- Store the Probability for each level and class.
        5.- Format message.
        6.- Produce message.

        Parameters
        ----------
        messages : dict-like
            Current object data, it must have the features and object id.

        """
        self.logger.info("Processing %i messages.", len(messages))
        self.logger.info("Getting batch alert data")
        predictor_input = self.predictor_parser.parse_input(messages)
        self.logger.info("Doing inference")
        tree_probabilities = self.model.predict_in_pipeline(predictor_input.value)
        self.logger.info("Processing results")
        results = self.predictor_parser.parse_output(tree_probabilities)
        alert_data = pd.DataFrame(
            [
                {"aid": message.get("aid"), "candid": message.get("candid", np.nan)}
                for message in messages
            ]
        )
        return {
            "public_info": (alert_data, predictor_input.value, tree_probabilities),
            "db_results": results,
        }

    def post_execute(self, result: dict):
        db_results = result.pop("db_results")
        self.produce_scribe(db_results.classifications)
        return result["public_info"]
