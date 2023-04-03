from apf.core.step import GenericStep
from db_plugins.db.mongo.connection import DatabaseConnection
from survey_parser_plugins import ALeRCEParser
from typing import List

from .utils import wizard, parser
import logging
import pandas as pd


class SortingHatStep(GenericStep):
    def __init__(
        self,
        db_connection: DatabaseConnection,
        config: dict,
        level=logging.INFO,
        **kwargs,
    ):
        super().__init__(config=config, level=level, **kwargs)
        self.driver = db_connection
        self.driver.connect(config["DB_CONFIG"])
        self.parser = ALeRCEParser()

    def pre_produce(self, result: pd.DataFrame):
        """
        Step lifecycle method.
        Format output that will be taken by the producer and set the producer key.

        Calling self.set_producer_key_field("aid") will make that each message produced has the
        aid value as key.

        For example if message looks like this:

        message = {"aid": "ALabc123"}

        Then, the produced kafka message will have "ALabc123" as key.

        Parameters
        ----------
        result: pd.DataFrame
            Data returned by the execute method

        Returns
        -------
        output_result: pd.DataFrame
            The parsed data as defined by the config["PRODUCER_CONFIG"]["SCHEMA"]
        """
        output_result = [parser.parse_output(alert) for _, alert in result.iterrows()]
        n_messages = len(output_result)
        self.set_producer_key_field("aid")
        self.logger.info(f"{n_messages} messages to be produced")
        return output_result

    def _add_metrics(self, alerts: pd.DataFrame):
        self.metrics["ra"] = alerts["ra"].tolist()
        self.metrics["dec"] = alerts["dec"].tolist()
        self.metrics["oid"] = alerts["oid"].tolist()
        self.metrics["tid"] = alerts["tid"].tolist()
        self.metrics["aid"] = alerts["aid"].tolist()

    def execute(self, messages: List[dict]):
        """
        Execute method of APF. Consumes message from CONSUMER_SETTINGS.
        :param messages: List of deserialized messages
        :return: Dataframe with the alerts
        """
        response = self.parser.parse(messages)
        alerts = pd.DataFrame(response)
        self.logger.info(f"Processing {len(alerts)} alerts")
        # Put name of ALeRCE in alerts
        alerts = self.add_aid(alerts)
        self._add_metrics(alerts)
        return alerts

    def add_aid(self, alerts: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an alerce_id to a batch of alerts given its oid, ra, dec and radius.
        :param alerts: Dataframe of alerts
        :return: Dataframe of alerts with a new column called `aid` (alerce_id)
        """
        # Internal cross-match that identifies same objects in own batch: create a new column named 'tmp_id'
        alerts = wizard.internal_cross_match(alerts)
        # Interaction with database: group all alerts with the same tmp_id and find/create alerce_id
        alerts = wizard.find_existing_id(self.driver, alerts)
        alerts = wizard.find_id_by_conesearch(self.driver, alerts)
        alerts = wizard.generate_new_id(alerts)
        alerts.drop(columns=["tmp_id"], inplace=True)
        return alerts
