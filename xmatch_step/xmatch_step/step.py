import json
import time
from typing import Dict, List, Tuple
import pandas as pd
from apf.core import get_class
from apf.core.step import GenericStep
from xmatch_step.core.xmatch_client import XmatchClient
from xmatch_step.core.utils.constants import ALLWISE_MAP
from xmatch_step.core.utils.extract_info import (
    extract_detections_from_messages,
)
from xmatch_step.core.parsing import parse_output


class XmatchStep(GenericStep):
    def __init__(
        self,
        config=None,
        **step_args,
    ):
        super().__init__(config=config, **step_args)

        self.xmatch_config = config["XMATCH_CONFIG"]
        self.xmatch_client = XmatchClient()
        self.catalog = self.xmatch_config["CATALOG"]

        cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])

        # Xmatch client config
        self.retries = config["RETRIES"]
        self.retry_interval = config["RETRY_INTERVAL"]

    @property
    def xmatch_parameters(self):
        return {
            "catalog_alias": self.catalog["name"],
            "columns": self.catalog["columns"],
            "radius": 1,
            "selection": "best",
            "input_type": "pandas",
            "output_type": "pandas",
        }

    def produce_scribe(self, xmatches: pd.DataFrame, oid_hash: Dict):
        if len(xmatches) == 0:
            return

        result = xmatches.rename(ALLWISE_MAP, axis="columns")
        result["catid"] = "allwise"
        allwise = result.drop(["dist"], axis=1)
        allwise = allwise.to_dict(orient="records")
        result.rename(columns={"oid_catalog": "catoid", "aid_in": "aid"}, inplace=True)

        data = result[["aid", "catoid", "dist", "catid"]]
        object_list = data.to_dict(orient="records")

        for obj in object_list:
            aid = obj.pop("aid")
            scribe_data = {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": aid, "oid": oid_hash[aid]},
                "data": {"xmatch": obj, "allwise": allwise},
            }
            self.scribe_producer.produce({"payload": json.dumps(scribe_data)})

    def pre_produce(self, result: List[dict]):
        def _add_non_detection(message):
            message["non_detections"] = (
                [] if message["non_detections"] is None else message["non_detections"]
            )
            return message

        self.set_producer_key_field("aid")
        return list(map(_add_non_detection, result))

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
                    self.xmatch_parameters["input_type"],
                    self.xmatch_parameters["catalog_alias"],
                    self.xmatch_parameters["columns"],
                    self.xmatch_parameters["selection"],
                    self.xmatch_parameters["output_type"],
                    self.xmatch_parameters["radius"],
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

    @classmethod
    def get_last_oid(cls, light_curves: pd.DataFrame):
        def _get_oid(series: pd.Series):
            candid = series["candid"]
            for det in series["detections"]:
                if str(det["candid"]) == str(candid):
                    return det["oid"]

        oid = light_curves.apply(_get_oid, axis=1)
        return oid

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
        self.logger.info(f"Processing {len(messages)} light curves")

        lc_hash = extract_detections_from_messages(messages)

        light_curves = pd.DataFrame.from_records(
            messages, exclude=["detections", "non_detections"]
        )
        light_curves.drop_duplicates(["aid"], keep="last", inplace=True)

        input_catalog = light_curves[["aid", "meanra", "meandec"]]

        if len(input_catalog) == 0:
            return [], pd.DataFrame.from_records([])

        # rename columns of meanra and meandec to (ra, dec)
        input_catalog.rename(columns={"meanra": "ra", "meandec": "dec"}, inplace=True)

        if self.config.get("FEATURE_FLAGS", {}).get("SKIP_XMATCH", False):
            self.logger.info("Skip xmatches")
            xmatches = pd.DataFrame(columns=["ra_in", "dec_in", "col1", "aid_in"])
        else:
            self.logger.info("Getting xmatches")
            xmatches = self.request_xmatch(input_catalog, self.retries)
        # Get output format
        candids = {}
        for msg in messages:
            if msg["aid"] not in candids:
                candids[msg["aid"]] = []
            candids[msg["aid"]].extend(msg["candid"])
        output_messages = parse_output(light_curves, xmatches, lc_hash, candids)
        del messages
        del light_curves
        del input_catalog
        return output_messages, xmatches, {k: v["oid"] for k, v in lc_hash.items()}

    def post_execute(self, result: Tuple[List[dict], pd.DataFrame, Dict]):
        self.produce_scribe(result[1], result[2])
        return result[0]
