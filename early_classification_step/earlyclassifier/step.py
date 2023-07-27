import sys
import os
import requests
import logging
import datetime
import pandas as pd
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from db_plugins.db.sql.models import Object, Probability, Step
from db_plugins.db.sql.connection import SQLConnection
from model.deployment import StampClassifier

DIRNAME = os.path.dirname(__file__)
MODEL = os.path.join(DIRNAME, "../model")
sys.path.append(MODEL)

FULL_ASTEROID_PROBABILITY = {
    "AGN": 0,
    "SN": 0,
    "bogus": 0,
    "asteroid": 1,
    "VS": 0,
}


class EarlyClassifier(GenericStep):
    """EarlyClassifier Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(
        self,
        consumer=None,
        config=None,
        db_connection=None,
        request_session=None,
        stamp_classifier=None,
        producer=None,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(consumer, config=config, level=level)
        self.db = db_connection or SQLConnection()
        self.db.connect(self.config["DB_CONFIG"]["SQL"])
        self.requests_session = request_session or requests.Session()
        self.model = stamp_classifier or StampClassifier()
        self.producer = None
        if config.get("PRODUCER_CONFIG", False) or producer is not None:
            self.producer = producer or KafkaProducer(config=config["PRODUCER_CONFIG"])

        if not step_args.get("test_mode", False):
            self.insert_step_metadata()

    def insert_step_metadata(self) -> None:
        self.db.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_ID"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["STEP_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )

    def is_asteroid(self, message: dict) -> bool:
        """
        Verifies if the candidate is an asteroid using conditions reported by the survey
        Parameters:
            message: dict
                The whole alert from the stream that has the `candidate` key.

        Returns:
            bool
                False if ssdistnr is -999.0 else true

        """
        # return ssdistnr != -999.0 and
        return message["candidate"]["ssdistnr"] != -999.0

    def sn_must_be_saved(self, message: dict, probabilities: dict) -> bool:
        if max(probabilities, key=probabilities.get) == "SN":
            msg = f"Object {message['objectId']} is classified as SN. "
            candidate = message["candidate"]
            if candidate["isdiffpos"] in ["f", 0]:
                msg += "But has a ifdiffpos positive"
                self.logger.info(msg)
                return False
            if candidate["sgscore1"] > 0.5 and candidate["distpsnr1"] < 1:
                msg += "But is near a star"
                self.logger.info(msg)
                return False
        return True

    def get_probabilities(self, message: dict) -> dict:
        if self.is_asteroid(message):
            return FULL_ASTEROID_PROBABILITY.copy()
        oid = message["objectId"]
        template = message["cutoutTemplate"]["stampData"]
        science = message["cutoutScience"]["stampData"]
        difference = message["cutoutDifference"]["stampData"]
        df = pd.DataFrame(
            [
                {
                    "oid": oid,
                    "cutoutScience": science,
                    "cutoutTemplate": template,
                    "cutoutDifference": difference,
                    **message["candidate"],
                }
            ],
            index=[oid],
        )
        try:
            probabilities = self.model.execute(df).iloc[0].to_dict()
        except Exception as e:
            self.logger.critical(str(e))
            probabilities = None

        if probabilities is not None and not self.sn_must_be_saved(
            message, probabilities
        ):
            probabilities = None
        return probabilities

    def insert_db(self, probs, oid, object_data) -> None:
        """
        Inserts probabilities returned by the stam classifier into the database.

        Parameters
        ----------
        probabilities : dict
            Should contain the probability name as key and probability as value.
            Something like:
            {
                "AGN": 1,
                "SN": 2,
                "bogus": 3,
                "asteroid": 4,
                "VS": 5,
            }
        oid : str
            Object ID of the object that is being classified. It is used to search
            the database for a specific object and get its instance
        object_data : dict
            Values to use if the object needs to be created. Keys in this dictionary
            should have every attribute of the object table other than `oid`
        """
        self.logger.info(probs)
        probabilities = self.get_ranking(probs)
        obj, _ = self.db.query(Object).get_or_create(
            filter_by={"oid": oid}, **object_data
        )

        for prob in probabilities:
            filter_by = {
                "oid": oid,
                "class_name": prob,
                "classifier_name": self.config["STEP_METADATA"]["CLASSIFIER_NAME"],
                "classifier_version": self.config["STEP_METADATA"][
                    "CLASSIFIER_VERSION"
                ],
            }
            prob_data = {
                "probability": probabilities[prob]["probability"],
                "ranking": probabilities[prob]["ranking"],
            }
            probability, created = self.db.query(Probability).get_or_create(
                filter_by=filter_by,
                **prob_data,
            )
            if not created:
                self.logger.info("Probability already exists. Skipping insert")
                break

    def get_ranking(self, probabilities) -> dict:
        """
        Transforms the probabilities dictionary returned by the model to a dictionary
        that has the probability and ranking for each class.

        Parameters
        ----------
        probabilities : dict
            Something like:
            {
                "AGN": 1,
                "SN": 2,
                "bogus": 3,
                "asteroid": 4,
                "VS": 5,
            }
        Return
        ------
        probabilities : dict
            A dictionary with probability and ranking for each class.
            Something like:
            {
                "AGN": {"probability": 1, "ranking": 5},
                "SN": {"probability": 2, "ranking": 4},
                "bogus": {"probability": 3, "ranking": 3},
                "asteroid": {"probability": 4, "ranking": 2},
                "VS": {"probability": 5, "ranking": 1},
            }

        """
        sorted_classes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        probs_with_ranking = {}
        for x in range(len(sorted_classes)):
            probs_with_ranking[sorted_classes[x][0]] = {
                "probability": probabilities[sorted_classes[x][0]],
                "ranking": x + 1,
            }
        return probs_with_ranking

    def get_default_object_values(
        self,
        alert: dict,
    ) -> dict:
        """
        Returns default values for creating an `object` in the database

        Parameters
        ----------
        alert : dict
            The whole alert from the stream that has the `candidate` key.

        Return
        ------
        data : dict
            Dictionary with default values for an object
        """
        data = {
            "ndethist": alert["candidate"]["ndethist"],
            "ncovhist": alert["candidate"]["ncovhist"],
            "mjdstarthist": alert["candidate"]["jdstarthist"] - 2400000.5,
            "mjdendhist": alert["candidate"]["jdendhist"] - 2400000.5,
            "firstmjd": alert["candidate"]["jd"] - 2400000.5,
            "ndet": 1,
            "deltajd": 0,
            "meanra": alert["candidate"]["ra"],
            "meandec": alert["candidate"]["dec"],
            "step_id_corr": "0.0.0",
            "corrected": False,
            "stellar": False,
        }
        data["lastmjd"] = data["firstmjd"]
        return data

    def produce(self, objectId, candid, probabilities):
        output = {}
        output["objectId"] = objectId
        output["candid"] = candid
        output["probabilities"] = probabilities
        self.producer.produce(output, key=objectId)

    def execute(self, message: dict) -> None:
        """
        Do model inference and insert of results in database

        Parameters
        ----------
        message : dict
            The whole alert from the stream that has the `candidate` key.
        Return
        ------
        None
        """
        oid = message["objectId"]
        candid = message["candidate"]["candid"]
        probabilities = self.get_probabilities(message)
        if probabilities is not None:
            object_data = self.get_default_object_values(message)
            self.insert_db(probabilities, oid, object_data)

            if self.producer:
                self.produce(oid, candid, probabilities)
