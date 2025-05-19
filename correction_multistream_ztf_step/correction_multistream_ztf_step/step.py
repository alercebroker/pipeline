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
        self.set_producer_key_field("oid")

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

        return {
            "detections": detections,
            "non_detections": non_detections.to_dict("records"),
            "coords": coords,
            "measurement_ids": measurement_ids,
        }

    @classmethod
    def pre_produce(cls, result: dict):
        result["detections"] = pd.DataFrame(result["detections"]).groupby("oid")

        try:  # At least one non-detection
            result["non_detections"] = pd.DataFrame(result["non_detections"]).groupby("oid")
        except KeyError:  # to reproduce expected error for missing non-detections in loop
            result["non_detections"] = pd.DataFrame(columns=["oid"]).groupby("oid")
        output = []

        for oid, dets in result["detections"]:

            dets = dets.replace(
                {np.nan: None, pd.NA: None, -np.inf: None}
            )  # Avoid NaN in the final results or infinite
            for field in [
                "e_ra",
                "e_dec",
            ]:  # Replace the e_ra/e_dec converted to None back to float nan per avsc formatting
                dets[field] = dets[field].apply(lambda x: x if pd.notna(x) else float("nan"))
            unique_measurement_ids = result["measurement_ids"][oid]
            unique_measurement_ids_long = [int(id_str) for id_str in unique_measurement_ids]

            detections_result = dets.to_dict("records")

            # Force the detection' parent candid back to integer
            for detections in detections_result:
                detections["measurement_id"] = int(detections["measurement_id"])
                parent_candid = detections.get("parent_candid")
                if parent_candid is not None and pd.notna(parent_candid):
                    detections["parent_candid"] = int(parent_candid)
                else:
                    detections["parent_candid"] = None

            output_message = {
                "oid": oid,
                "measurement_id": unique_measurement_ids_long,
                "meanra": result["coords"][oid]["meanra"],
                "meandec": result["coords"][oid]["meandec"],
                "detections": detections_result,
            }

            try:
                output_message["non_detections"] = (
                    result["non_detections"].get_group(oid).to_dict("records")
                )
            except KeyError:
                output_message["non_detections"] = []
            output.append(output_message)
        return output

    """

    def post_execute(self, result: dict):
        self.produce_scribe(result["detections"])
        return result

    def produce_scribe(self, detections: list[dict]):
        count = 0
        for detection in detections:
            count += 1
            flush = False
            # Prevent further modification for next step
            detection = deepcopy(detection)
            if not detection.pop("new"):
                continue
            measurement_id = detection.pop("measurement_id")
            oid = detection.get("oid")
            is_forced = detection.pop("forced")
            set_on_insert = not detection.get("has_stamp", False)
            extra_fields = detection["extra_fields"]
            # remove possible elasticc extrafields
            for to_remove in ["prvDiaSources", "prvDiaForcedSources", "fp_hists"]:
                extra_fields.pop(to_remove, None)
            if "diaObject" in extra_fields:
                extra_fields["diaObject"] = pickle.loads(extra_fields["diaObject"])
            detection["extra_fields"] = extra_fields
            scribe_data = {
                "collection": "forced_photometry" if is_forced else "detection",
                "type": "update",
                "criteria": {"measurement_id": measurement_id, "oid": oid},
                "data": detection,
                "options": {"upsert": True, "set_on_insert": set_on_insert},
            }
            scribe_payload = {"payload": json.dumps(scribe_data)}
            if count == len(detections):
                flush = True
            self.scribe_producer.produce(scribe_payload, flush=flush, key=oid)

    """

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
