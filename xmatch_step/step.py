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
    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)

        self.xmatch_config = config["XMATCH_CONFIG"]
        self.xmatch_client = XmatchClient()

        if "CLASS" in config["PRODUCER_CONFIG"]:
            Producer = get_class(config["PRODUCER_CONFIG"]["CLASS"])
        else:
            Producer = KafkaProducer
        self.producer = Producer(config["PRODUCER_CONFIG"])
        self.driver = SQLConnection()
        self.driver.connect(config["DB_CONFIG"]["SQL"])
        self.version = config["STEP_VERSION"]
        self.logger.info(f"XMATCH {self.version}")

    def _extract_coordinates(self, message: dict):
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

    def cast_allwise(self, wise_match: dict) -> dict:
        return {
            key: wise_match[key] for key in ALLWISE_KEYS if key in wise_match.keys()
        }

    def cast_xmatch(self, match: dict) -> dict:
        return {key: match[key] for key in XMATCH_KEYS if key in match.keys()}

    def save_xmatch(self, result):
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

        result["catid"] = "allwise"
        res_arr = result.to_dict(orient="records")

        self.driver.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_ID"]},
            name= self.config["STEP_NAME"],
            version=self.config["STEP_VERSION"],
            comments=self.config["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )

        for d in res_arr:
            filter = {"oid": d["oid"]}
            object, created = self.driver.session.query().get_or_create(
                Object, filter_by=filter
            )
            filter = {"oid_catalog": d["oid_catalog"]}
            data = self.cast_allwise(d)
            allwise, created = self.driver.session.query().get_or_create(
                Allwise, filter_by=filter, **data
            )

            filter = {"oid": d["oid"]}
            data = self.cast_xmatch(d)
            xmatch, created = self.driver.session.query().get_or_create(
                Xmatch, filter_by=filter, **data
            )


    def _produce(self, messages):
        for message in messages:
            self.producer.produce(message)

    def execute(self, messages):
        array = []
        for m in messages:
            record = self._extract_coordinates(m)
            array.append(record)

        df = pd.DataFrame(array, columns=["oid", "ra", "dec"])

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

        self.save_xmatch(result)
        self.driver.session.commit()

        messages = self._format_result(messages, df, result)

        self._produce(messages)
