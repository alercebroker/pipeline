import numpy as np
from apf.core import get_class
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from cds_xmatch_client import XmatchClient
from db_plugins.db.sql.models import Object, Allwise, Xmatch, Step
from db_plugins.db.sql import SQLConnection
from typing import List
from xmatch_step.utils.constants import XMATCH_KEYS, ALLWISE_KEYS
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
        # if not step_args.get("test_mode", False):
        #     self.insert_step_metadata()

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

    def _format_result(self, light_curves: pd.DataFrame, xmatches: pd.DataFrame) -> List[dict]:
        """ Join xmatches with input lightcurves. If xmatch not exists for an object, the value is None. Also generate
        a list of dict as output.
        :param light_curves: Generic messages that contain the light curves (in dataframe)
        :param xmatches: Values of cross-matches (in dataframe)
        :return:
        """
        # Create a new dataframe that contain just two columns `aid` and `xmatches`.
        aid_in = xmatches["aid_in"]
        xmatches.drop(columns=["ra_in", "dec_in", "col1", "aid_in"], inplace=True)
        xmatches.replace({np.nan: None}, inplace=True)
        xmatches = pd.DataFrame({
            "aid_in": aid_in,
            "xmatches": xmatches.apply(lambda x: x.to_dict(), axis=1)
        })
        # Join xmatches with light curves
        data = light_curves.set_index("aid").join(xmatches.set_index("aid_in"))
        data.replace({np.nan: None}, inplace=True)
        data.reset_index(inplace=True)
        # Transform to a list of dicts
        data = data.to_dict("records")
        return data

    def save_xmatch(self, result, df_object):
        if len(result) > 0:
            result = result.rename(
                {
                    "AllWISE": "oid_catalog",
                    "RAJ2000": "ra",
                    "DEJ2000": "dec",
                    "W1mag": "w1mpro",
                    "W2mag": "w2mpro",
                    "W3mag": "w3mpro",
                    "W4mag": "w4mpro",
                    "e_W1mag": "w1sigmpro",
                    "e_W2mag": "w2sigmpro",
                    "e_W3mag": "w3sigmpro",
                    "e_W4mag": "w4sigmpro",
                    "Jmag": "j_m_2mass",
                    "Hmag": "h_m_2mass",
                    "Kmag": "k_m_2mass",
                    "e_Jmag": "j_msig_2mass",
                    "e_Hmag": "h_msig_2mass",
                    "e_Kmag": "k_msig_2mass",
                    "oid_in": "oid",
                    "angDist": "dist",
                },
                axis="columns",
            )
            object_oid = df_object.oid
            result_oid = result.oid

            object_data = df_object[object_oid.isin(result_oid)]

            # bulk insert to object table
            array = object_data.to_dict(orient="records")
            self.driver.query(Object).bulk_insert(array)

            # bulk insert to allwise table
            data = result.drop(["oid", "dist"], axis=1)
            array = data.to_dict(orient="records")
            self.driver.query(Allwise).bulk_insert(array)

            # bulk insert to xmatch table
            result["catid"] = "allwise"
            data = result[["oid", "oid_catalog", "dist", "catid"]]
            array = data.to_dict(orient="records")
            self.driver.query(Xmatch).bulk_insert(array)

    def produce(self, messages: List[dict]) -> None:
        for message in messages:
            self.producer.produce(message, key=message["aid"])

    def request_xmatch(self, input_catalog: pd.DataFrame, retries_count: int) -> pd.DataFrame:
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
        light_curves.to_pickle("/home/javier/Desktop/DRs/data.pkl")
        light_curves.drop_duplicates(["aid", "candid"], keep="last", inplace=True)

        input_catalog = light_curves[["aid", "meanra", "meandec"]]
        input_catalog.rename(columns={"meanra": "ra", "meandec": "dec"}, inplace=True)

        # executing xmatch request
        self.logger.info(f"Getting xmatches")
        result = self.request_xmatch(input_catalog, self.retries)

        # Write in database
        self.logger.info(f"Writing xmatches in DB")
        # self.save_xmatch(result, object_df) # PSQL

        output_messages = self._format_result(light_curves, result)
        self.logger.info(f"Producing {len(output_messages)} messages")
        self.produce(output_messages)
