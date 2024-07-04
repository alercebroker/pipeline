from apf.core.step import GenericStep
from survey_parser_plugins import ALeRCEParser
from datetime import datetime
from typing import List

from .utils import wizard, parser
from .database import MongoConnection, PsqlConnection
import pandas as pd


class SortingHatStep(GenericStep):
    def __init__(
        self,
        mongo_connection: MongoConnection,
        config: dict,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.mongo_driver = mongo_connection
        self.parser = ALeRCEParser()
        # feature flags
        self.use_psql = config["FEATURE_FLAGS"]["USE_PSQL"]
        self.use_mongo = config["FEATURE_FLAGS"]["USE_MONGO"]
        self.run_conesearch = config["FEATURE_FLAGS"]["RUN_CONESEARCH"]

    def set_psql_driver(self, psql_connection: PsqlConnection):
        if not self.use_psql:
            raise ValueError("Cannot set psql driver when USE_PSQL is False")
        self.psql_driver = psql_connection

    def pre_produce(self, result: pd.DataFrame):
        """
        Step lifecycle method.
        Format output that will be taken by the producer and set the producer key.

        Calling self.set_producer_key_field("oid") will make that each message produced has the
        oid value as key.

        For example if message looks like this:

        message = {"oid": "abc123"}

        Then, the produced kafka message will have "abc123" as key.

        Parameters
        ----------
        result: pd.DataFrame
            Data returned by the execute method

        Returns
        -------
        output_result: pd.DataFrame
            The parsed data as defined by the config["PRODUCER_CONFIG"]["SCHEMA"]
        """
        output_result = [
            parser.parse_output(alert) for _, alert in result.iterrows()
        ]
        self.set_producer_key_field("oid")
        return output_result

    def _add_metrics(self, alerts: pd.DataFrame):
        self.metrics["ra"] = alerts["ra"].tolist()
        self.metrics["dec"] = alerts["dec"].tolist()
        self.metrics["oid"] = alerts["oid"].tolist()
        self.metrics["tid"] = alerts["tid"].tolist()
        self.metrics["aid"] = alerts["aid"].tolist()

    def pre_execute(self, messages: List[dict]):
        ingestion_timestamp = int(datetime.now().timestamp())
        messages_with_timestamp = list(
            map(
                lambda m: {
                    **m,
                    "brokerIngestTimestamp": ingestion_timestamp,
                    "surveyPublishTimestamp": m["timestamp"],
                    "source": m.get("topic", ""),
                },
                messages,
            )
        )
        return messages_with_timestamp

    def execute(self, messages: List[dict]):
        """
        Execute method of APF. Consumes message from CONSUMER_SETTINGS.
        :param messages: List of deserialized messages
        :return: Dataframe with the alerts
        """
        response = self.parser.parse(messages)
        alerts = pd.DataFrame([r.to_dict() for r in response])
        self.logger.info(f"Processing {len(alerts)} alerts")
        # Put name of ALeRCE in alerts
        alerts = self.add_aid(alerts)
        self._add_metrics(alerts)

        return alerts

    def post_execute(self, alerts: pd.DataFrame):
        """
        Writes entries to the database with _id and oid only.
        :param alerts: Dataframe of alerts
        """
        psql_driver = None
        if self.use_psql:
            psql_driver = self.psql_driver
        mongo_driver = None
        if self.use_mongo:
            mongo_driver = self.mongo_driver
        wizard.insert_empty_objects(mongo_driver, alerts, psql=psql_driver)
        return alerts

    def add_aid(self, alerts: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an alerce_id to a batch of alerts given its oid, ra, dec and radius.
        :param alerts: Dataframe of alerts
        :return: Dataframe of alerts with a new column called `aid` (alerce_id)
        """
        self.logger.info(f"Assigning AID to {len(alerts)} alerts")
        alerts["aid"] = None
        # Interaction with database: group all alerts with the same oid and find/create alerce_id
        if self.use_mongo:
            alerts = wizard.find_existing_id(self.mongo_driver, alerts)
            if self.run_conesearch:
                alerts = wizard.find_id_by_conesearch(
                    self.mongo_driver, alerts
                )
        alerts = wizard.generate_new_id(alerts)
        return alerts
