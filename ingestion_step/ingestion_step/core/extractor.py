from typing import Any, Iterable

import pandas as pd

from ingestion_step.core.types import DType, Message


class BaseExtractor:
    schema: dict[str, DType] = {}
    extra_columns_schema: dict[str, DType] = {}
    field: str = ""

    @classmethod
    def _extra_columns(
        cls,
        _message: Message,
        _measurements: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        return {}

    @classmethod
    def extract(cls, messages: Iterable[Message]) -> pd.DataFrame:
        schema = (
            cls.schema
            | cls.extra_columns_schema
            | {"message_id": pd.Int32Dtype()}
        )
        data = {col: [] for col in schema}

        for message_id, message in enumerate(messages):
            measurements: list[Any] = []
            if type(message[cls.field]) is list:
                measurements = message[cls.field]
            elif message[cls.field] is not None:
                measurements = [message[cls.field]]

            for measurement in measurements:
                for col in cls.schema:
                    data[col].append(measurement[col])
            data["message_id"] += [message_id] * len(measurements)

            extra_columns = cls._extra_columns(message, measurements)
            for name, col in extra_columns.items():
                data[name] += col

        return pd.DataFrame(
            {
                col: pd.Series(data[col], dtype=dtype)
                for col, dtype in schema.items()
            }
        )
