from apf.core.step import GenericStep
from apf.core import get_class
from apf.producers import KafkaProducer
from db_plugins.db.mongo.connection import MongoDatabaseCreator
from db_plugins.db.generic import new_DBConnection
from survey_parser_plugins import ALeRCEParser
from typing import List
from .utils.sorting_hat import SortingHat

import numpy as np
import pandas as pd
import pickle
import logging


class SortingHatStep(GenericStep):
    def __init__(
        self,
        consumer=None,
        config=None,
        level=logging.INFO,
        producer=None,
        db_connection=None,
        **step_args,
    ):
        super().__init__(consumer, config=config, level=level)
        if not producer and config.get("PRODUCER_CONFIG", False):
            if "CLASS" in config["PRODUCER_CONFIG"]:
                producer_class = get_class(config["PRODUCER_CONFIG"]["CLASS"])
                producer = producer_class(config["PRODUCER_CONFIG"])
            elif "PARAMS" in config["PRODUCER_CONFIG"]:
                producer = KafkaProducer(config["PRODUCER_CONFIG"])

        self.producer = producer
        self.driver = db_connection or new_DBConnection(MongoDatabaseCreator)
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
        alerts = alerts.replace(
            {np.nan: None}
        )  # transform np.nan to None (only for produce proposals)
        for index, alert in alerts.iterrows():
            alert = alert.to_dict()
            alert["rfid"] = None if alert["rfid"] is None else int(alert["rfid"])
            for k in alert[
                "extra_fields"
            ].keys():  # transform to bytes if datatype is list
                if isinstance(alert["extra_fields"][k], list):
                    alert["extra_fields"][k] = pickle.dumps(alert["extra_fields"][k])
            # produce alert content with key of candid
            self.producer.produce(alert, key=str(alert["candid"]))
            n_messages += 1
        self.logger.info(f"{n_messages} messages Produced")

    def execute(self, messages: List[dict]) -> None:
        """
        Execute method of APF. This method consume message from CONSUMER_SETTINGS.
        :param messages: List of deserialized messages
        :return:
        """
        response = self.parser.parse(messages)
        alerts = pd.DataFrame(response)
        del alerts["stamps"]
        self.logger.info(f"Processing {len(alerts)} alerts")
        # Put name of ALeRCE in alerts
        alerts = self.wizard.to_name(alerts)
        if self.producer:
            self.produce(alerts)
        del alerts
        del messages
        del response
