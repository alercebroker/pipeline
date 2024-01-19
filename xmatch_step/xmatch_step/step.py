import json
import time
from typing import Dict, List, Tuple

import pandas as pd
from apf.consumers import KafkaConsumer
from apf.core import get_class
from apf.core.step import GenericStep

from xmatch_step.core.parsing import parse_output
from xmatch_step.core.utils.constants import ALLWISE_MAP
from xmatch_step.core.utils.extract_info import (
    extract_lightcurve_from_messages,
    get_candids_from_messages,
)
from xmatch_step.core.xmatch_client import XmatchClient


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

    def pre_execute(self, messages: List[dict]):
        def remove_timestamp(message: dict):
            message.pop("timestamp", None)
            return message

        messages = list(map(remove_timestamp, messages))
        return messages

    def produce_scribe(self, xmatches: pd.DataFrame):
        if len(xmatches) == 0:
            return
        result = xmatches.rename(ALLWISE_MAP, axis="columns")
        result["catid"] = "allwise"
        allwise = result.drop(["dist"], axis=1)
        allwise = allwise.to_dict(orient="records")
        result.rename(
            columns={"oid_catalog": "catoid", "oid_in": "oid"}, inplace=True
        )
        result = result[["oid", "catoid", "dist", "catid"]]
        result = result.to_dict(orient="records")
        flush = False
        for idx, obj in enumerate(result):
            oid = obj.pop("oid")
            obj = {"allwise": {"catoid": obj["catoid"], "dist": obj["dist"]}}
            scribe_data = {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": oid},
                "data": {"xmatch": obj},
            }
            if idx == len(result) - 1:
                flush = True
            self.scribe_producer.produce(
                {"payload": json.dumps(scribe_data)}, flush=flush
            )

    def pre_produce(self, result: Tuple[pd.DataFrame, Dict, Dict]):
        self.set_producer_key_field("oid")
        xmatches, lightcurves_by_oid, candids = result
        output_messages = parse_output(xmatches, lightcurves_by_oid, candids)
        return output_messages

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
                self.logger.warning(
                    f"CDS xmatch client returned with error {e}"
                )
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
        """Perform xmatch against CDS."""
        self.logger.info(f"Processing {len(messages)} light curves")
        xmatches = pd.DataFrame(columns=["ra_in", "dec_in", "col1", "aid_in"])

        lightcurves_by_oid = extract_lightcurve_from_messages(messages)
        candids = get_candids_from_messages(messages)

        messages_df = pd.DataFrame.from_records(
            messages, exclude=["detections", "non_detections"]
        ).drop_duplicates(["oid"], keep="last")

        input_catalog = messages_df[["oid", "meanra", "meandec"]].rename(
            columns={"meanra": "ra", "meandec": "dec"}
        )
        if (
            not self.config.get("FEATURE_FLAGS", {}).get("SKIP_XMATCH", False)
            and len(input_catalog) > 0
        ):
            self.logger.info("Getting xmatches")
            xmatches = self.request_xmatch(input_catalog, self.retries)

        return xmatches, lightcurves_by_oid, candids

    def post_execute(self, result: Tuple[pd.DataFrame, Dict, Dict]):
        xmatches, lightcurves_by_oid, candids = result
        self.produce_scribe(xmatches)
        return xmatches, lightcurves_by_oid, candids

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
