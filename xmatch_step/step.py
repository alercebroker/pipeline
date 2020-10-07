from apf.core import get_class
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from cds_xmatch_client import XmatchClient
from db_plugins.db.sql.models import Object, Allwise, Xmatch, Step

import pandas as pd
import logging
import datetime
import os

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

        if "CLASS" in config["PRODUCER_CONFIG"]:
            Producer = get_class(config["PRODUCER_CONFIG"]["CLASS"])
        else:
            Producer = KafkaProducer
        self.producer = producer or Producer(config["PRODUCER_CONFIG"])
        self.driver = db_connection or SQLConnection()
        self.driver.connect(config["DB_CONFIG"]["SQL"])
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.logger.info(f"XMATCH {self.version}")
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
        Get ra, dec and oid from alert message.

        Poarameters
        -----------
        message : dict
            ztf alert message from stream

        Returns
        -------
        record : dict
            a dict with oid, ra and dec keys
        """
        record = {
            "oid": message["objectId"],
            "ra": message["candidate"]["ra"],
            "dec": message["candidate"]["dec"],
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

        object_data = df_object[df_object.oid.isin(result.oid)]

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

    def get_default_object_values(
        self,
        alert: dict,
    ) -> dict:

        data = {"oid": alert["objectId"]}
        data["ndethist"] = alert["candidate"]["ndethist"]
        data["ncovhist"] = alert["candidate"]["ncovhist"]
        data["mjdstarthist"] = alert["candidate"]["jdstarthist"] - 2400000.5
        data["mjdendhist"] = alert["candidate"]["jdendhist"] - 2400000.5
        data["firstmjd"] = alert["candidate"]["jd"] - 2400000.5
        data["lastmjd"] = data["firstmjd"]
        data["ndet"] = 1
        data["deltajd"] = 0
        data["meanra"] = alert["candidate"]["ra"]
        data["meandec"] = alert["candidate"]["dec"]
        data["step_id_corr"] = "0.0.0"
        data["corrected"] = False
        data["stellar"] = False

        return data

    def _produce(self, messages):
        for message in messages:
            self.producer.produce(message, key=message["objectId"])

    def execute(self, messages):
        array = []
        object_array = []
        for m in messages:
            record = self._extract_coordinates(m)
            array.append(record)

            record = self.get_default_object_values(m)
            object_array.append(record)

        df = pd.DataFrame(array, columns=["oid", "ra", "dec"])
        object_df = pd.DataFrame(object_array)

        # xmatch
        catalog = self.xmatch_config["CATALOG"]
        catalog_alias = catalog["name"]
        columns = catalog["columns"]
        radius = 1
        selection = "best"
        input_type = "pandas"
        output_type = "pandas"

        result = self.xmatch_client.execute(
            df, input_type, catalog_alias, columns, selection, output_type, radius
        )

        # Write in database
        self.save_xmatch(result, object_df)

        messages = self._format_result(messages, df, result)
        self._produce(messages)
