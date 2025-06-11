from datetime import datetime
from typing import Any

import pandas as pd
from apf.core.step import GenericStep
from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.parser_interface import ParsedData
from ingestion_step.core.select_parser import select_parser
from ingestion_step.utils.database import (
    insert_detections,
    insert_forced_photometry,
    insert_non_detections,
    insert_objects,
)
from ingestion_step.ztf.serializer import serialize_ztf


class IngestionStep(GenericStep):
    ingestion_timestamp: int | None

    def __init__(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)
        self.parser = select_parser(config["SURVEY_STRATEGY"])
        self.psql_driver = PsqlDatabase(config["PSQL_CONFIG"])

    def _add_metrics(self, alerts: pd.DataFrame):
        self.metrics: dict[str, Any] = {}
        self.metrics["ra"] = alerts["ra"].tolist()
        self.metrics["dec"] = alerts["dec"].tolist()
        self.metrics["oid"] = alerts["oid"].tolist()
        self.metrics["tid"] = alerts["tid"].tolist()
        self.metrics["aid"] = alerts["aid"].tolist()

    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, messages: list[dict[str, Any]]
    ) -> ParsedData:
        self.logger.info(f"Processing {len(messages)} alerts")

        self.ingestion_timestamp = int(datetime.now().timestamp())
        parsed_data = self.parser.parse(messages)

        self.logger.info(f'Parsed {len(parsed_data["objects"])=}')
        self.logger.info(f'Parsed {len(parsed_data["detections"])=}')
        self.logger.info(f'Parsed {len(parsed_data["non_detections"])=}')
        self.logger.info(f'Parsed {len(parsed_data["forced_photometries"])=}')

        insert_objects(self.psql_driver, parsed_data["objects"])
        insert_detections(self.psql_driver, parsed_data["detections"])
        insert_non_detections(self.psql_driver, parsed_data["non_detections"])
        insert_forced_photometry(
            self.psql_driver, parsed_data["forced_photometries"]
        )

        return parsed_data

    def pre_produce(
        self, result: ParsedData
    ):  # pyright: ignore[reportIncompatibleMethodOverride]
        self.set_producer_key_field("oid")
        messages = serialize_ztf(result)

        return messages
