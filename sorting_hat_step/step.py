from apf.consumers import GenericConsumer
from apf.core.step import GenericStep
from apf.producers import GenericProducer
from db_plugins.db.mongo.connection import DatabaseConnection
from survey_parser_plugins import ALeRCEParser
from typing import List

from sorting_hat_step.utils.database import db_queries
from .utils import sorting_hat as wizard, output as output_parser
import logging
import pandas as pd
from operator import itemgetter


class SortingHatStep(GenericStep):
    def __init__(
        self,
        consumer: GenericConsumer,
        config: dict,
        producer: GenericProducer,
        db_connection: DatabaseConnection,
        level=logging.INFO,
    ):
        super().__init__(consumer, config=config, level=level)
        self.producer = producer
        self.driver = db_connection
        self.driver.connect(config["DB_CONFIG"])
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.parser = ALeRCEParser()
        self.oid_query, self.conesearch_query = itemgetter(
            "oid_query", "conesearch_query"
        )(db_queries(self.driver))

    def produce(self, alerts: pd.DataFrame) -> None:
        """
        Produce generic alerts to producer with configuration of PRODUCER_CONFIG from settings.py.
        :param alerts: Dataframe of generic alerts with alerce_id
        :return:
        """
        n_messages = 0
        for _, alert in alerts.iterrows():
            output = output_parser.parse_output(alert)
            self.producer.produce(output, key=str(output["aid"]))
            n_messages += 1
        self.logger.info(f"{n_messages} messages Produced")

    def _add_metrics(self, alerts: pd.DataFrame):
        self.metrics["ra"] = alerts["ra"].tolist()
        self.metrics["dec"] = alerts["dec"].tolist()
        self.metrics["oid"] = alerts["oid"].tolist()
        self.metrics["tid"] = alerts["tid"].tolist()
        self.metrics["aid"] = alerts["aid"].tolist()

    def execute(self, messages: List[dict]) -> None:
        """
        Execute method of APF. This method consume message from CONSUMER_SETTINGS.
        :param messages: List of deserialized messages
        :return:
        """
        response = self.parser.parse(messages)
        alerts = pd.DataFrame(response)
        self.logger.info(f"Processing {len(alerts)} alerts")
        # Put name of ALeRCE in alerts
        alerts = self.add_aid(alerts)
        self._add_metrics(alerts)
        if self.producer:
            self.produce(alerts)
        del alerts
        del messages
        del response

    def add_aid(self, alerts: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an alerce_id to a batch of alerts given its oid, ra, dec and radius.
        :param alerts: Dataframe of alerts
        :return: Dataframe of alerts with a new column called `aid` (alerce_id)
        """
        # Internal cross-match that identifies same objects in own batch: create a new column named 'tmp_id'
        alerts = wizard.internal_cross_match(alerts)
        # Interaction with database: group all alerts with the same tmp_id and find/create alerce_id
        alerts = wizard.find_existing_id(alerts, self.oid_query)
        alerts = wizard.find_id_by_conesearch(alerts, self.conesearch_query)
        alerts = wizard.generate_new_id(alerts)
        alerts.drop(columns=["tmp_id"])
        return alerts
