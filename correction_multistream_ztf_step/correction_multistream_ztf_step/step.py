import logging
import numpy as np
import pandas as pd
import json
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
from core.parsers.parser_utils import NumpyEncoder, parse_output_df, parse_data_for_avro
from core.parsers.scribe_parser import scribe_parser
from core.schemas import schema_applier
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
        self.scribe_enabled = config.get("SCRIBE_ENABLED", True)
        if self.scribe_enabled:
            cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
            self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])

    def execute(self, messages: List[dict]) -> dict:
        all_detections = []
        all_non_detections = []
        msg_data = []
        print("AAAAAAAAAAAAAAAAAAAAAAAAA")
        for msg in messages:
            oid = msg["oid"]
            measurement_id = msg["measurement_id"]
            msg_data.append({"oid": oid, "measurement_id": measurement_id})

            parent_candid_list = []
            for detection in msg["detections"]:
                parsed_detection = detection.copy()
                parsed_detection["oid"] = oid
                parsed_detection["new"] = True
                parent_candid_list.append(parsed_detection["parent_candid"])

                all_detections.append(parsed_detection)

            for non_detection in msg["non_detections"]:
                parsed_non_detection = non_detection.copy()
                parsed_non_detection["oid"] = oid
                all_non_detections.append(parsed_non_detection)

        msg_df = pd.DataFrame(msg_data)

        # Separate all detections back into candidate, forced_phot and prv_candidates
        # This is so we can recreate the pandas dataframe using the schemas for each type of detection without losing any precision at all
        # In the future this will be dropped once we send them separate from ingestion step, and simplify this because extra fields will not exist
        from core.schemas.message_candidate_schema import candidate_schema

        schema_msg_candidate = candidate_schema

        candidates = [detection for detection in all_detections if detection["has_stamp"] is True]

        df_candidates = schema_applier.apply_schema_flatten_data(candidates, schema_msg_candidate)

        from core.schemas.message_prvcandidate_schema import prv_candidate_schema

        schema_msg_prvcandidate = prv_candidate_schema

        prv_candidates = [
            detection
            for detection in all_detections
            if detection["forced"] is not True and detection["parent_candid"] is not None
        ]

        df_prv_candidates = schema_applier.apply_schema_flatten_data(
            prv_candidates, schema_msg_prvcandidate
        )

        from core.schemas.message_fp_schema import forced_photometry_schema

        schema_msg_fp = forced_photometry_schema

        forced_photometry = [
            detection for detection in all_detections if detection["forced"] is True
        ]

        df_forced_photometry = schema_applier.apply_schema_flatten_data(
            forced_photometry, schema_msg_fp
        )

        # Join them together back into a single df
        # We can process each of them each in a separate Corrector, but for now it will be done this way
        # By processing each separated, its simple to change it so we can send to the topic correction ms ztf only the necessary fields for what comes next afterwards
        detections_df = pd.concat([df_candidates, df_forced_photometry, df_prv_candidates])

        from core.schemas.db_sql_ndet_schema import db_sql_non_detection_schema

        schema_db_sql_non_det = db_sql_non_detection_schema

        if all_non_detections:
            non_detections_df = schema_applier.apply_schema_flatten_data(
                all_non_detections, schema_db_sql_non_det
            )

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

        """Queries the database for all detections and non-detections for each OID and removes duplicates"""
        db_sql_detections = _get_sql_detections(oids, self.db_sql, parse_sql_detection)
        db_sql_non_detections = _get_sql_non_detections(oids, self.db_sql, parse_sql_non_detection)
        db_sql_forced_photometries = _get_sql_forced_photometries(
            oids, self.db_sql, parse_sql_forced_photometry
        )

        from core.schemas.db_sql_schema import db_sql_detection_schema

        schema_db_sql_det = db_sql_detection_schema
        db_sql_detections_df = schema_applier.apply_schema_flatten_data(
            db_sql_detections, schema_db_sql_det
        )

        from core.schemas.db_sql_fphot_schema import schema_db_sql_fphot

        schema_db_sql_fphot = schema_db_sql_fphot
        db_sql_forced_photometries_df = schema_applier.apply_schema_flatten_data(
            db_sql_forced_photometries, schema_db_sql_fphot
        )

        # Join all the detections and sql detections into a single dataframe (Will be skipped in the future to process each type separate)
        detections = pd.concat([detections_df, db_sql_detections_df, db_sql_forced_photometries_df])

        # The same schema used to parse original non detections is compatible with ndets here
        db_sql_non_detections_df = schema_applier.apply_schema_flatten_data(
            db_sql_non_detections, schema_db_sql_non_det
        )
        non_detections = pd.concat([db_sql_non_detections_df, non_detections_df])

        self.logger.debug(f"Retrieved {detections.shape[0]} detections")
        detections["measurement_id"] = detections["measurement_id"].astype(str)

        # has_stamp true will be on top
        # new true will be on top
        detections = detections.sort_values(["has_stamp", "new"], ascending=[False, False])

        # so this will drop alerts coming from the database if they are also in the stream
        # but will also drop if they are previous detections
        detections = detections.drop_duplicates(["measurement_id", "oid"], keep="first")
        non_detections = non_detections.drop_duplicates(["oid", "band", "mjd"])
        self.logger.debug(f"Obtained {len(detections[detections['new']])} new detections")

        non_detections = (
            non_detections if not non_detections.empty else pd.DataFrame(columns=["oid"])
        )

        if not self.config["FEATURE_FLAGS"].get("SKIP_MJD_FILTER", False):
            detections = detections[detections["mjd"] <= detections["oid"].map(last_mjds)]

        corrector = Corrector(detections)
        detections_df = corrector.corrected_as_dataframe()
        coords_df = corrector.coordinates_as_dataframe()
        non_detections = non_detections.drop_duplicates(["oid", "band", "mjd"])

        parsed_output = parse_output_df(detections_df, non_detections, coords_df, measurement_ids)
        clean_output = parse_data_for_avro(parsed_output)
        return clean_output

    def post_execute(self, result: dict):
        self.produce_scribe(scribe_parser(result))
        return result

    def produce_scribe(self, scribe_payloads):
        for scribe_data in scribe_payloads:
            payload = {"payload": json.dumps(scribe_data, cls=NumpyEncoder)}
            self.scribe_producer.produce(payload)

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
