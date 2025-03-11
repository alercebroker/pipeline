from datetime import datetime
from typing import Any

import pandas as pd
from apf.core.step import GenericStep
from ingestion_step.parser.core.parser_interface import ParserInterface
from ingestion_step.parser.select_parser import select_parser

from settings import StepConfig

from .database import PsqlConnection
from .utils import parser, wizard


class SortingHatStep(GenericStep):
    ingestion_timestamp: int | None

    def __init__(
        self,
        config: StepConfig,
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)  # pyright: ignore
        self.parser = select_parser(config["SURVEY_STRATEGY"])
        self.psql_driver = PsqlConnection(config["PSQL_CONFIG"])

    def _add_metrics(self, alerts: pd.DataFrame):
        self.metrics: dict[str, Any] = {}
        self.metrics["ra"] = alerts["ra"].tolist()
        self.metrics["dec"] = alerts["dec"].tolist()
        self.metrics["oid"] = alerts["oid"].tolist()
        self.metrics["tid"] = alerts["tid"].tolist()
        self.metrics["aid"] = alerts["aid"].tolist()

    def execute(self, messages: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Execute method of APF. Consumes message from CONSUMER_SETTINGS.
        :param messages: list of deserialized messages
        :return: Dataframe with the alerts
        """
        self.ingestion_timestamp = int(datetime.now().timestamp())
        self.parser.parse(messages)

        common_obj = self.parser.get_common_objects()

        # No he actualizado el codigo bajo esta linea

        self.logger.info(f"Processing {len(alerts)} alerts")
        self._add_metrics(alerts)

        return alerts

    def post_execute(self, alerts: pd.DataFrame):
        """
        Writes entries to the database with _id and oid only.
        :param alerts: Dataframe of alerts
        """
        psql_driver = self.psql_driver
        wizard.insert_empty_objects(psql_driver, alerts)
        return alerts

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
        output_result = [parser.parse_output(alert) for _, alert in result.iterrows()]
        self.set_producer_key_field("oid")
        return output_result
