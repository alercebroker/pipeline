from apf.core import get_class
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from lc_classifier.classifier.models import HierarchicalRandomForest
from lc_classification.utils import NoClassifiedPostProcessor

import logging
import pandas as pd
import numpy as np
import json

import numexpr


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

        self.features_required = set(self.model.feature_list)

        cls = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(config["SCRIBE_PRODUCER_CONFIG"])

    def get_ranking(self, df):
        ranking = (-df).rank(axis=1, method="dense", ascending=True).astype(int)
        return ranking

    def stack_df(self, df, ranking):
        df = df.stack()
        ranking = ranking.stack()
        df.rename("probability", inplace=True)
        ranking.rename("ranking", inplace=True)
        result = pd.concat([df, ranking], axis=1)
        result.index.names = ["aid", "class_name"]
        result.reset_index(inplace=True)
        return result

    def get_classifier_name(self, suffix=None):
        return self.base_name if suffix is None else f"{self.base_name}_{suffix}"

    def process_results(self, tree_probabilities):
        probabilities = tree_probabilities["probabilities"]
        top = tree_probabilities["hierarchical"]["top"]
        children = tree_probabilities["hierarchical"]["children"]

        top_ranking = self.get_ranking(top)
        probabilities_ranking = self.get_ranking(probabilities)

        top_result = self.stack_df(top, top_ranking)
        probabilities_result = self.stack_df(probabilities, probabilities_ranking)

        probabilities_result["classifier_name"] = self.get_classifier_name()
        top_result["classifier_name"] = self.get_classifier_name("top")

        results = [top_result, probabilities_result]
        for key in children:
            child_ranking = self.get_ranking(children[key])
            child_result = self.stack_df(children[key], child_ranking)
            child_result["classifier_name"] = self.get_classifier_name(key.lower())
            results.append(child_result)

        results = pd.concat(results)
        results["classifier_version"] = self.model.MODEL_VERSION_NAME
        return results

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

    def produce_scribe(self, db_results: pd.DataFrame):
        db_results.set_index("aid")
        # TODO: get schema of the operation
        classifications = db_results.to_dict("records")
        for classification in classifications:
            aid = classification.pop("aid")
            command = {
                "collection": "object",
                "type": "update_probabilities",
                "criteria": {"_id": aid},
                "data": classification,
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
        alert_data = self.message_to_df(messages)
        features = self.features_to_df(alert_data, messages)
        self.logger.info("Found %i Features.", len(features))
        missing_features = self.features_required.difference(set(features.columns))

        if len(missing_features) > 0:
            raise KeyError(
                f"Corrupted Batch: missing some features ({missing_features})"
            )
        self.logger.info("Doing inference")
        features.set_index("aid", inplace=True)
        tree_probabilities = self.model.predict_in_pipeline(features)
        self.logger.info("Processing results")
        db_results = self.process_results(tree_probabilities)

        return {
            "public_info": (alert_data, features, tree_probabilities),
            "db_results": db_results,
        }

    def post_execute(self, result: dict):
        db_results = result.pop("db_results")
        self.produce_scribe(db_results)
        return result["public_info"]
