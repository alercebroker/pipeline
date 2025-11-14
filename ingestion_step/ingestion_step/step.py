from typing import Any

import pandas as pd
from apf.core.step import GenericStep
from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.select_parser import select_parser
from ingestion_step.core.strategy import ParsedData
from ingestion_step.core.types import Message


class IngestionStep(GenericStep):
    ingestion_timestamp: int | None

    def __init__(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)
        self.Strategy = select_parser(config["SURVEY_STRATEGY"])
        self.psql_driver = PsqlDatabase(config["PSQL_CONFIG"])
        self.insert_batch_size = config.get("INSERT_BATCH_SIZE")
        self.producer.set_key_field("oid")


    def _add_metrics(self, alerts: pd.DataFrame):
        self.metrics: dict[str, Any] = {}
        self.metrics["ra"] = alerts["ra"].tolist()
        self.metrics["dec"] = alerts["dec"].tolist()
        self.metrics["oid"] = alerts["oid"].tolist()
        self.metrics["tid"] = alerts["tid"].tolist()
        self.metrics["aid"] = alerts["aid"].tolist()

    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, messages: list[Message]
    ) -> ParsedData:
        parsed_data = self.Strategy.parse(messages)

        for key in parsed_data:
            self.logger.info(f"Parsed {len(parsed_data[key])} objects form {key}")

        self.Strategy.insert_into_db(
            self.psql_driver, parsed_data, chunk_size=self.insert_batch_size
        )

        return parsed_data

    def pre_produce(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, result: ParsedData
    ):
        self.set_producer_key_field(self.Strategy.get_key())
        messages = self.Strategy.serialize(result)

        return messages
