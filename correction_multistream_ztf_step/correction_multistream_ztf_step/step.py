import logging
import numpy as np
import pandas as pd
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer
from typing import List

### Imports for lightcurve database queries
from core.DB.database_sql import (
    PSQLConnection,
    _get_sql_detections,
    _get_sql_forced_photometries,
    _get_sql_non_detections,
)
from core.parsers.parser_sql import (
    parse_sql_detection,
    parse_sql_forced_photometry,
    parse_sql_non_detection,
)

from apf.core import get_class

from core.parsers.parser_utils import parse_output

from core.parsers.scribe_parser import scribe_parser


from core.corrector import Corrector



class CorrectionMultistreamZTFStep(GenericStep):
    def __init__(
        self,
        config: dict,
        db_sql: PSQLConnection,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.CorrectionMultistreamZTFStep")
        self.scribe_enabled = config.get("SCRIBE_ENABLED", False)
        if self.scribe_enabled:
            cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
            self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])

    def execute(self, messages: List[dict]) -> dict:
        all_detections = []
        all_non_detections = []
        msg_data = []

        for msg in messages:
            oid = msg["oid"]
            measurement_id = msg["measurement_id"]
            msg_data.append({"oid": oid, "measurement_id": measurement_id})

            for detection in msg["detections"]:
                parsed_detection = detection.copy()
                parsed_detection["oid"] = oid
                parsed_detection["new"] = True
                keys_to_remove = [
                    "magpsf",
                    "magpsf_corr",
                    "sigmapsf",
                    "sigmapsf_corr",
                    "sigmapsf_corr_ext",
                ]

                #   Remove the keys from the extra_fields dictionary (sigmapsf is duped with mag_corr and it doesnt make sense having that data in the message)
                for key in keys_to_remove:
                    if key in parsed_detection["extra_fields"]:
                        parsed_detection["extra_fields"].pop(key, None)

                all_detections.append(parsed_detection)

            for non_detection in msg["non_detections"]:
                parsed_non_detection = non_detection.copy()
                parsed_non_detection["oid"] = oid
                all_non_detections.append(parsed_non_detection)

        msg_df = pd.DataFrame(msg_data)
        detections_df = pd.DataFrame(
            all_detections
        )  # We will always have detections BUT not always non_detections
        # Keep the parent candid as int instead of scientific notation
        detections_df["parent_candid"] = detections_df["parent_candid"].astype("Int64")
        # Tranform the NA parent candid to None
        detections_df["parent_candid"] = (
            detections_df["parent_candid"]
            .astype(object)
            .where(~detections_df["parent_candid"].isna(), None)
        )
        if all_non_detections:
            non_detections_df = pd.DataFrame(all_non_detections)
        else:
            non_detections_df = pd.DataFrame(
                columns=["oid", "measurement_id", "band", "mjd", "diffmaglim"]
            )

        oids = set(msg_df["oid"].unique())

        measurement_ids = (
            msg_df.groupby("oid")["measurement_id"].apply(lambda x: [str(id) for id in x]).to_dict()
        )
        last_mjds = detections_df.groupby("oid")["mjd"].max().to_dict()

        logger = logging.getLogger("alerce.CorrectionMultistreamZTFStep")
        logger.debug(f"Received {len(detections_df)} detections from messages")
        oids = list(oids)

        detections = detections_df.to_dict("records")
        non_detections = non_detections_df.to_dict("records")
        """Queries the database for all detections and non-detections for each OID and removes duplicates"""
        db_sql_detections = _get_sql_detections(oids, self.db_sql, parse_sql_detection)
        db_sql_non_detections = _get_sql_non_detections(oids, self.db_sql, parse_sql_non_detection)
        db_sql_forced_photometries = _get_sql_forced_photometries(
            oids, self.db_sql, parse_sql_forced_photometry
        )

        detections = pd.DataFrame(detections + db_sql_detections + db_sql_forced_photometries)
        non_detections = pd.DataFrame(non_detections + db_sql_non_detections)

        self.logger.debug(f"Retrieved {detections.shape[0]} detections")
        detections["measurement_id"] = detections["measurement_id"].astype(str)

        # TODO: check if this logic is ok
        # TODO: has_stamp in db is not reliable
        # has_stamp true will be on top
        # new true will be on top
        detections = detections.sort_values(["has_stamp", "new"], ascending=[False, False])

        # so this will drop alerts coming from the database if they are also in the stream
        # but will also drop if they are previous detections
        detections = detections.drop_duplicates(["measurement_id", "oid"], keep="first")
        non_detections = non_detections.drop_duplicates(["oid", "band", "mjd"])
        self.logger.debug(f"Obtained {len(detections[detections['new']])} new detections")

        non_detections = (
            non_detections.replace(np.nan, None)
            if not non_detections.empty
            else pd.DataFrame(columns=["oid"])
        )

        if not self.config["FEATURE_FLAGS"].get("SKIP_MJD_FILTER", False):
            detections = detections[detections["mjd"] <= detections["oid"].map(last_mjds)]

        corrector = Corrector(detections)
        detections = corrector.corrected_as_records()
        non_detections = non_detections.replace({float("nan"): None})
        coords = corrector.coordinates_as_records()
        non_detections = non_detections.drop_duplicates(["oid", "band", "mjd"])

        result = {
            "detections": detections,
            "non_detections": non_detections.to_dict("records"),
            "coords": coords,
            "measurement_ids": measurement_ids,
        }

        parsed_output = parse_output(result)
        return parsed_output

    def post_execute(self, result: dict):

        #self.produce_scribe(scribe_parser(result))

        return result           
    
    """
    def produce_scribe(self, detections: list[dict]):
            self.scribe_producer.produce(scribe_payload, flush=flush, key=oid)

    """

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
