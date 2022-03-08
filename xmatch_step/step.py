from apf.core import get_class
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from cds_xmatch_client import XmatchClient
from db_plugins.db.sql.models import Object, Allwise, Xmatch, Step

import pandas as pd
import logging
import datetime
import os
import time
from db_plugins.db.sql import SQLConnection

ALLWISE_KEYS = [
    "oid_catalog",
    "ra",
    "dec",
    "w1mpro",
    "w2mpro",
    "w3mpro",
    "w4mpro",
    "w1sigmpro",
    "w2sigmpro",
    "w3sigmpro",
    "w4sigmpro",
    "j_m_2mass",
    "h_m_2mass",
    "k_m_2mass",
    "j_msig_2mass",
    "h_msig_2mass",
    "k_msig_2mass",
]
XMATCH_KEYS = ["oid", "catid", "oid_catalog", "dist", "class_catalog", "period"]


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
        if not step_args.get("test_mode", False):
            self.insert_step_metadata()

    def insert_step_metadata(self):
        """
        Inserts step metadata like version and step name.
        Some config is required:

        .. code-block:: python

            #settings.py

            STEP_CONFIG = {
                STEP_METADATA = {
                    "STEP_VERSION": os.getenv("STEP_VERSION", "dev"),
                    "STEP_ID": os.getenv("STEP_VERSION", "dev"),
                    "STEP_NAME": os.getenv("STEP_VERSION", "dev"),
                    "STEP_COMMENTS": "",
                }
            }

        """
        self.driver.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_ID"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["STEP_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )

    def _extract_coordinates(self, message: dict):
        """
        Get meanra, meandec and oid from alert message.

        Poarameters
        -----------
        message : dict
            alerce alert message from stream

        Returns
        -------
        record : dict
            a dict with oid, ra and dec keys
        """
        record = {
            "candid": message["candid"],
            "oid": message["oid"],
            "ra": message["candidate"]["meanra"],
            "dec": message["candidate"]["meandec"],
        }
        return record

    def _format_result(self, msgs, input, result):
        messages = []
        # objects without xmatch
        without_result = input[~input["oid_in"].isin(result["oid_in"])]["oid_in"].values

        for m in msgs:
            oid = m["objectId"]
            if oid in without_result:
                m["xmatches"] = None
            else:
                sel = result[result["oid_in"] == oid]
                row = sel.iloc[0]
                columns = dict(row)
                del columns["oid_in"]
                m["xmatches"] = {"allwise": columns}
            messages.append(m)

        return messages

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

    def _produce(self, messages):
        for message in messages:
            self.producer.produce(message, key=message["objectId"])

    def convert_null_to_none(self, columns, d):
        for column in columns:
            if d.get(column) == "null":
                d[column] = None

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} alerts")
        array = []

        for m in messages:
            self.convert_null_to_none(["drb"], m["candidate"])
            record = self._extract_coordinates(m)
            array.append(record)

        df = pd.DataFrame(array, columns=["candid", "oid", "ra", "dec"])

        df.drop_duplicates(subset=["candid"], inplace=True)

        # executing xmatch request
        self.logger.info(f"Getting xmatches")
        result = self.request_xmacth_result_with_retries(df, self.retries)

        # Write in database
        self.logger.info(f"Writing xmatches in DB")
        self.save_xmatch(result, object_df)

        messages = self._format_result(messages, df, result)
        self.logger.info(f"Producing messages")
        self._produce(messages)

    def request_xmacth_result_with_retries(self, data_frame, retries_count):
        if retries_count > 0:
            try:
                # make request
                result = self.xmatch_client.execute(
                    data_frame,
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
                # error catched sleep before retry
                time.sleep(self.retry_interval)
                self.logger.warning("Retrying request")
                return self.request_xmacth_result_with_retries(data_frame, retries_count - 1)

        if retries_count == 0:
            # raise error coundt find xmatch
            self.logger.error(
                f"Retrieving xmatch from the client unsuccessful after {self.retries} retries. Shutting down."
            )
            raise Exception(
                f"Could not retrieve xmatch from CDS after {self.retries} retries."
            )
