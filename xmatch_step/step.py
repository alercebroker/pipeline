from apf.core import get_class
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from cds_xmatch_client import XmatchClient
from db_plugins.db.sql.models import Allwise, Xmatch, Step
from db_plugins.db.sql import SQLConnection
from typing import List, Set
from xmatch_step.utils.constants import ALLWISE_MAP

import numpy as np
import pandas as pd
import logging
import datetime
import time


class XmatchStep(GenericStep):
    def __init__(
        self,
        consumer=None,
        config=None,
        level=logging.INFO,
        db_connection=None,
        xmatch_client=None,
        producer=None,
        **step_args,
    ):
        super().__init__(consumer, config=config, level=level)

        self.xmatch_config = config["XMATCH_CONFIG"]
        self.xmatch_client = xmatch_client or XmatchClient()

        # xmatch variables
        self.catalog = self.xmatch_config["CATALOG"]
        self.catalog_alias = self.catalog["name"]
        self.columns = self.catalog["columns"]
        self.radius = 1
        self.selection = "best"
        self.input_type = "pandas"
        self.output_type = "pandas"

        if "CLASS" in config["PRODUCER_CONFIG"]:
            Producer = get_class(config["PRODUCER_CONFIG"]["CLASS"])
        else:
            Producer = KafkaProducer
        self.producer = producer or Producer(config["PRODUCER_CONFIG"])
        self.driver = db_connection or SQLConnection()
        self.driver.connect(config["DB_CONFIG"]["SQL"])
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.logger.info(f"XMATCH {self.version}")
        self.retries = config["RETRIES"]
        self.retry_interval = config["RETRY_INTERVAL"]

    def insert_step_metadata(self):
        """
        Inserts step metadata like version and step name.
        """
        self.driver.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_ID"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["STEP_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )

    def _format_result(
        self, light_curves: pd.DataFrame, xmatches: pd.DataFrame
    ) -> List[dict]:
        """Join xmatches with input lightcurves. If xmatch not exists for an object, the value is None. Also generate
        a list of dict as output.
        :param light_curves: Generic messages that contain the light curves (in dataframe)
        :param xmatches: Values of cross-matches (in dataframe)
        :return:
        """
        # Create a new dataframe that contain just two columns `aid` and `xmatches`.
        aid_in = xmatches["aid_in"]
        # Temporal code: the oid_in will be removed
        xmatches.drop(columns=["ra_in", "dec_in", "col1", "oid_in", "aid_in"], inplace=True)
        xmatches.replace({np.nan: None}, inplace=True)
        xmatches = pd.DataFrame(
            {
                "aid_in": aid_in,
                "xmatches": xmatches.apply(lambda x: None if x is None else {"allwise": x.to_dict()}, axis=1),
            }
        )
        # Join xmatches with light curves
        data = light_curves.set_index("aid").join(xmatches.set_index("aid_in"))
        data.replace({np.nan: None}, inplace=True)
        data.index.name = "aid"
        data.reset_index(inplace=True)
        # Transform to a list of dicts
        data = data.to_dict("records")
        return data

    def save_xmatch(self, xmatches: pd.DataFrame):
        if len(xmatches) > 0:
            result = xmatches.rename(
                ALLWISE_MAP,
                axis="columns",
            )
            self.logger.info(f"Writing xmatches in DB")
            # bulk insert to allwise table
            data = result.drop(["dist"], axis=1)
            array = data.to_dict(orient="records")
            self.driver.query(Allwise).bulk_insert(array)

            # bulk insert to xmatch table
            result["catid"] = "allwise"
            data = result[["oid", "oid_catalog", "dist", "catid"]]
            array = data.to_dict(orient="records")
            self.driver.query(Xmatch).bulk_insert(array)

    def produce(self, messages: List[dict]) -> None:
        def exists_in_ztf(oids: Set[str]) -> bool:
            for o in oids:
                if "ZTF" in o:
                    return True
            return False

        for message in messages:
            # Temporal code: for produce only ZTF objects:
            if exists_in_ztf(message["oid"]):
                del message["oid"]
                self.producer.produce(message, key=message["aid"])

    def request_xmatch(
        self, input_catalog: pd.DataFrame, retries_count: int
    ) -> pd.DataFrame:
        """
        Recursive method allows request CDS xmatch.
        :param input_catalog: Input catalog in dataframe. Must contain columns: ra, dec, identifier.
        :param retries_count: number of attempts
        :return: Data with its xmatch. Data without xmatch is not included.
        """
        if retries_count > 0:
            try:
                result = self.xmatch_client.execute(
                    input_catalog,
                    self.input_type,
                    self.catalog_alias,
                    self.columns,
                    self.selection,
                    self.output_type,
                    self.radius,
                )
                return result

            except Exception as e:
                self.logger.warning(f"CDS xmatch client returned with error {e}")
                time.sleep(self.retry_interval)
                self.logger.warning("Retrying request")
                return self.request_xmatch(input_catalog, retries_count - 1)

        if retries_count == 0:
            self.logger.error(
                f"Retrieving xmatch from the client unsuccessful after {self.retries} retries. Shutting down."
            )
            raise Exception(
                f"Could not retrieve xmatch from CDS after {self.retries} retries."
            )

    def execute(self, messages: List[dict]) -> None:
        """
        Execute method. Contains the logic of the xmatch step, it does the following:
        - Parse messages to a dataframe (and remove duplicated data)
        - Generate an input catalog
        - Do a xmatch between input catalog and CDS
        - Join matches with input message
        - Produce
        :param messages: Input messages from stream
        :return: None
        """
        self.logger.info(f"Processing {len(messages)} alerts")
        light_curves = pd.DataFrame(messages)
        light_curves.drop_duplicates(["aid", "candid"], keep="last", inplace=True)

        # Temporal code: to manage oids of ZTF and store xmatch
        light_curves["oid"] = light_curves["detections"].apply(lambda x: set(det["oid"] for det in x))
        input_catalog = light_curves[["aid", "meanra", "meandec", "oid"]]  # Temp. code: remove only oid
        input_catalog = input_catalog.explode("oid", ignore_index=True)
        mask_ztf = input_catalog["oid"].str.contains("ZTF")
        input_catalog = input_catalog[mask_ztf]
        # End of temporal code

        # rename columns of meanra and meandec to (ra, dec)
        input_catalog.rename(columns={"meanra": "ra", "meandec": "dec"}, inplace=True)
        # executing xmatch request
        self.logger.info(f"Getting xmatches")
        xmatches = self.request_xmatch(input_catalog, self.retries)
        # Write in database
        self.save_xmatch(xmatches)  # PSQL
        # Get output format
        output_messages = self._format_result(light_curves, xmatches)
        self.logger.info(f"Producing {len(output_messages)} messages")
        # Produce data with xmatch
        self.produce(output_messages)
        del messages
        del light_curves
        del input_catalog
        del xmatches
        del output_messages
