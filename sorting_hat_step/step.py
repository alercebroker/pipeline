from apf.consumers import GenericConsumer
from apf.core.step import GenericStep
from apf.producers import GenericProducer
from db_plugins.db.mongo.connection import DatabaseConnection
from survey_parser_plugins import ALeRCEParser
from typing import List
from .utils.sorting_hat import SortingHat
from .utils.output import _parse_output
import logging
import pandas as pd


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
        self.wizard = SortingHat(self.driver)

    def produce(self, alerts: pd.DataFrame) -> None:
        """
        Produce generic alerts to producer with configuration of PRODUCER_CONFIG from settings.py.
        :param alerts: Dataframe of generic alerts with alerce_id
        :return:
        """
        n_messages = 0
        for _, alert in alerts.iterrows():
            output = _parse_output(alert)
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
        alerts = self.wizard.to_name(alerts)
        self._add_metrics(alerts)
        if self.producer:
            self.produce(alerts)
        del alerts
        del messages
        del response
