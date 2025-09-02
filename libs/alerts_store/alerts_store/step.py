import logging
from pathlib import Path
from typing import Any

import fastavro
from apf.core.step import GenericStep
from fastavro.schema import load_schema
from fastavro.types import Schema

Message = dict[str, Any]


class AlertStore(GenericStep):
    base_folder: str
    logger: logging.Logger
    schema: Schema

    def __init__(self, config: dict[str, Any], **kwargs: Any):
        super().__init__(config=config, **kwargs)
        self.base_folder = config["BASE_FOLDER"]

        self.schema = load_schema(config["CONSUMER_CONFIG"]["SCHEMA_PATH"])

    def execute(self, messages: list[Message]):
        self.logger.info(f"Start processing {len(messages)} messages.")
        for i, message in enumerate(messages):
            dia_source_id: int = message["diaSourceId"]

            dia_source: dict[str, Any] = message["diaSource"]
            mjd: float = dia_source["midpointMjdTai"]

            folder_path = Path(self.base_folder) / str(int(mjd))
            folder_path.mkdir(parents=True, exist_ok=True)

            file_path = folder_path / f"alert_{dia_source_id}.avro"

            with open(file_path, "wb") as f:
                fastavro.schemaless_writer(f, self.schema, message)

        self.logger.info(f"Saved {len(messages)} messages.")

        return messages

    def produce(self, result: list[Message]):  # pyright: ignore
        pass
