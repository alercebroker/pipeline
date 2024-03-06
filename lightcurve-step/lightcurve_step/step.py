import logging
import numpy as np
import pandas as pd
import pickle
from typing import List
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer
from .database_mongo import (
    DatabaseConnection,
    _get_mongo_non_detections,
    _get_mongo_detections,
    _get_mongo_forced_photometries,
)
from .database_sql import (
    PSQLConnection,
    _get_sql_detections,
    _get_sql_forced_photometries,
    _get_sql_non_detections,
)
from .parser_mongo import (
    parse_mongo_forced_photometry,
    parse_mongo_detection,
    parse_mongo_non_detection,
)
from .parser_sql import (
    parse_sql_detection,
    parse_sql_forced_photometry,
    parse_sql_non_detection,
)


class LightcurveStep(GenericStep):
    def __init__(
        self,
        config: dict,
        db_mongo: DatabaseConnection,
        db_sql: PSQLConnection,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.db_mongo = db_mongo
        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.LightcurveStep")
        self.set_producer_key_field("oid")

    def pre_execute(self, messages: List[dict]) -> dict:
        detections, non_detections, oids = [], [], set()
        last_mjds = {}
        candids = {}
        for msg in messages:
            oid = msg["oid"]
            oids.add(oid)
            if oid not in candids:
                candids[oid] = []
            candids[oid].append(str(msg["candid"]))
            last_mjds[oid] = max(
                last_mjds.get(oid, 0), msg["detections"][0]["mjd"]
            )
            detections.extend(
                [det | {"new": True} for det in msg["detections"]]
            )
            non_detections.extend(msg["non_detections"])

        logger = logging.getLogger("alerce.LightcurveStep")
        logger.debug(f"Received {len(detections)} detections from messages")
        return {
            "oids": list(oids),
            "candids": candids,
            "last_mjds": last_mjds,
            "detections": detections,
            "non_detections": non_detections,
        }

    def execute(self, messages: dict) -> dict:
        """Queries the database for all detections and non-detections for each OID and removes duplicates"""

        db_mongo_detections = _get_mongo_detections(
            messages["oids"], self.db_mongo, parse_mongo_detection
        )
        db_mongo_non_detections = _get_mongo_non_detections(
            messages["oids"], self.db_mongo, parse_mongo_non_detection
        )
        db_mongo_forced_photometries = _get_mongo_forced_photometries(
            messages["oids"],
            self.db_mongo,
            parse_mongo_forced_photometry,
        )
        db_sql_detections = _get_sql_detections(
            messages["oids"], self.db_sql, parse_sql_detection
        )
        db_sql_non_detections = _get_sql_non_detections(
            messages["oids"], self.db_sql, parse_sql_non_detection
        )
        db_sql_forced_photometries = _get_sql_forced_photometries(
            messages["oids"], self.db_sql, parse_sql_forced_photometry
        )
        detections = pd.DataFrame(
            messages["detections"]
            + db_mongo_detections
            + db_sql_detections
            + db_mongo_forced_photometries
            + db_sql_forced_photometries
        )
        non_detections = pd.DataFrame(
            messages["non_detections"]
            + db_mongo_non_detections
            + db_sql_non_detections
        )
        self.logger.debug(f"Retrieved {detections.shape[0]} detections")
        detections["candid"] = detections["candid"].astype(str)
        detections["parent_candid"] = detections["parent_candid"].astype(str)

        # has_stamp true will be on top
        # new true will be on top
        # so this will drop alerts coming from the database if they are also in the stream
        # but will also drop if they are previous detections
        detections = detections.sort_values(
            ["has_stamp", "new"], ascending=[False, False]
        ).drop_duplicates(["candid", "oid"], keep="first")

        non_detections = non_detections.drop_duplicates(["oid", "fid", "mjd"])
        self.logger.debug(
            f"Obtained {len(detections[detections['new']])} new detections"
        )
        return {
            "candids": messages["candids"],
            "detections": detections,
            "non_detections": non_detections,
            "last_mjds": messages["last_mjds"],
        }

    def pre_produce(self, result: dict) -> List[dict]:
        def serialize_dia_object(ef: dict):
            if "diaObject" not in ef or not isinstance(
                ef.get("diaObject"), list
            ):
                return ef

            ef["diaObject"] = pickle.dumps(ef["diaObject"])
            return ef

        detections = result["detections"].replace(np.nan, None).groupby("oid")
        try:  # At least one non-detection
            non_detections = (
                result["non_detections"].replace(np.nan, None).groupby("oid")
            )
        except KeyError:  # Handle empty non-detections
            non_detections = pd.DataFrame(columns=["oid"]).groupby("oid")
        output = []
        for oid, dets in detections:
            try:
                nd = non_detections.get_group(oid).to_dict("records")
            except KeyError:
                nd = []
            dets["extra_fields"] = dets["extra_fields"].apply(
                serialize_dia_object
            )
            if not self.config["FEATURE_FLAGS"].get("SKIP_MJD_FILTER", False):
                mjds = result["last_mjds"]
                dets = dets[dets["mjd"] <= mjds[oid]]
            detections_list = dets.to_dict("records")

            output.append(
                {
                    "oid": oid,
                    "candid": result["candids"][oid],
                    "detections": detections_list,
                    "non_detections": nd,
                }
            )
        return output

    def tear_down(self):
        self.db_mongo.client.close()
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
