from apf.core import get_class
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from late_classifier.classifier.models import HierarchicalRandomForest, PICKLE_PATH

import logging
import pandas as pd
import scipy.stats as sstats
import numpy as np
import datetime

from db_plugins.db.sql import SQLConnection
from db_plugins.db.sql.models import (
    Object,
    Probability,
    Taxonomy,
    Step,
)

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

    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)

        numexpr.utils.set_num_threads(1)
        self.logger.info("Loading Models")
        self.model = HierarchicalRandomForest({})
        self.model.download_model()
        self.model.load_model(self.model.MODEL_PICKLE_PATH)
        self.features_required = set(self.model.feature_list)

        if "CLASS" in config["PRODUCER_CONFIG"]:
            Producer = get_class(config["PRODUCER_CONFIG"]["CLASS"])
        else:
            Producer = KafkaProducer

        self.producer = Producer(config["PRODUCER_CONFIG"])
        self.driver = SQLConnection()
        self.driver.connect(config["DB_CONFIG"]["SQL"])
        self.driver.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_ID"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["STEP_NAME"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )

    def _format_features(self, features):
        """Format a message that correspond features `dict`.
        This method take the keys and values, transform it to DataFrame. After that
        rename some columns for make this readable for classifier.
        **Example:**

        Parameters
        ----------
        message : dict
            Message deserialized of Kafka.
        """
        features = pd.DataFrame.from_records([features])
        return features

    def get_probability(
        self, oid, class_name, classifier, version, probability, ranking
    ):
        """Get single probability instace from the database.

        Parameters
        ----------
        oid : str
            Object unique identifier.
        class_name : str
            Class name.
        classifier : str
            Classifier name.
        version : str
            Classifier version.
        probability : float
            Probability asigned by the Classifier to the class.
        ranking : int
            Position relative to the biggest probability.

        Returns
        -------
            Probability, created
                The probability object itself and if was created or not.


        """
        filters = {
            "oid": oid,
            "class_name": class_name,
            "classifier_name": classifier,
            "classifier_version": version,
        }
        data = {"probability": probability, "ranking": ranking}
        return self.driver.session.query().get_or_create(
            Probability, filter_by=filters, **data
        )

    def set_taxonomy(self, classes, classifier_name, classifier_version):
        """Save the class taxonomy if it doesn't exists.

        Parameters
        ----------
        classes : list[array]
            List of classes.
        classifier_name : str
            Classifier Name.
        classifier_version : str
            Classifier Version.

        """
        filters = {
            "classifier_name": classifier_name,
            "classifier_version": classifier_version,
        }
        data = {"classes": classes}
        return self.driver.session.query().get_or_create(
            Taxonomy, filter_by=filters, **data
        )

    def get_ranking(self, probabilities):
        """Given a dictionary of probabilities, get the ranking relative to the biggest probability.

        Parameters
        ----------
        probabilities : dict
            Dictionary of `'class': probability` used to calculate the ranking.

        Returns
        -------
        list
            Array of positions.

        """
        values = probabilities.values()
        values = list(values)
        values = np.array(values)
        return sstats.rankdata(-values, method="dense")

    def insert_dict(self, oid, dictionary, suffix=None):
        """Insert and updates probabilities.

        Parameters
        ----------
        oid : str
            Object id.
        dictionary : dict
            Probabilities in a `'class': probability` format.
        suffix : str
            Suffix for the classifier_name (i.e. late_classifier[_top]).

        """
        probabilities = []
        classifier_name = (
            self.base_name if suffix is None else f"{self.base_name}_{suffix}"
        )
        ranking = self.get_ranking(dictionary)
        self.set_taxonomy(
            classes=list(dictionary.keys()),
            classifier_name=classifier_name,
            classifier_version=self.model.MODEL_VERSION_NAME,
        )
        for (class_name, probability, rank) in zip(
            dictionary.keys(), dictionary.values(), ranking
        ):
            probability = float(probability)
            rank = int(rank)
            prob, created = self.get_probability(
                oid=oid,
                class_name=class_name,
                classifier=classifier_name,
                version=self.model.MODEL_VERSION_NAME,
                probability=probability,
                ranking=rank,
            )
            prob.probability = probability
            prob.ranking = rank

            probabilities.append(prob)
        return probabilities

    def insert_db(self, result, oid):
        """Iterate over the late_classifier results and insert into the database.

        Parameters
        ----------
        result : dict
            Hierarchical tree of results.
        oid : str
            Object Id.

        Returns
        -------
        type
            Description of returned object.

        """
        probabilities = []
        final = result["probabilities"]
        prob_tmp = self.insert_dict(oid, final)
        probabilities.extend(prob_tmp)

        hierarchical = result["hierarchical"]
        top_probabilities = hierarchical["top"]
        prob_tmp = self.insert_dict(oid, top_probabilities, "top")
        probabilities.extend(prob_tmp)

        for child in hierarchical["children"]:
            child_probabilities = hierarchical["children"][child]
            prob_tmp = self.insert_dict(oid, child_probabilities, child.lower())
            probabilities.extend(prob_tmp)
        return probabilities

    def execute(self, message):
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
        message : dict-like
            Current object data, it must have the features and object id.

        """
        oid = message["oid"]
        features = self._format_features(message["features"])
        missing_features = self.features_required.difference(set(features.columns))

        if len(missing_features) != 0:
            self.logger.info(
                f"[{oid}] Missing {len(missing_features)} Features: {missing_features}"
            )
        else:
            self.logger.info(f"[{oid}] Processing")
            result = self.model.predict_in_pipeline(features)
            probabilities = self.insert_db(result, oid)

            new_message = {
                "oid": oid,
                "candid": message["candid"],
                "features": message["features"],
                "late_classification": result,
            }
            self.logger.info(f"[{oid}] Processed")
            self.producer.produce(new_message, key = oid)
            self.logger.info(f"[{oid}] Produced")
            self.driver.session.commit()
