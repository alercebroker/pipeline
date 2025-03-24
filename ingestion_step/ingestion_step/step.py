from datetime import datetime
from typing import Any, Hashable

import pandas as pd
from apf.core.step import GenericStep

from ingestion_step.core.parsed_data import ParsedData
from ingestion_step.core.select_parser import select_parser
from settings import StepConfig

from .database import PsqlConnection


class SortingHatStep(GenericStep):
    ingestion_timestamp: int | None

    def __init__(
        self,
        config: StepConfig,
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)  # pyright: ignore
        self.parser = select_parser(config["SURVEY_STRATEGY"])
        self.psql_driver = PsqlConnection(config["PSQL_CONFIG"])

    def _add_metrics(self, alerts: pd.DataFrame):
        self.metrics: dict[str, Any] = {}
        self.metrics["ra"] = alerts["ra"].tolist()
        self.metrics["dec"] = alerts["dec"].tolist()
        self.metrics["oid"] = alerts["oid"].tolist()
        self.metrics["tid"] = alerts["tid"].tolist()
        self.metrics["aid"] = alerts["aid"].tolist()

    def execute(self, messages: list[dict[str, Any]]) -> ParsedData:
        self.logger.info(f"Processing {len(messages)} alerts")

        self.ingestion_timestamp = int(datetime.now().timestamp())
        parsed_data = self.parser.parse(messages)

        self.logger.info(f'Parsed {len(parsed_data["objects"])=}')
        self.logger.info(f'Parsed {len(parsed_data["detections"])=}')
        self.logger.info(f'Parsed {len(parsed_data["non_detections"])=}')
        self.logger.info(f'Parsed {len(parsed_data["forced_photometries"])=}')

        return parsed_data

    def pre_produce(self, parsed_data: ParsedData) -> list[dict[Hashable, Any]]:
        detections = parsed_data["detections"].set_index("oid").to_dict("records")
        return detections
