from typing import Any, Iterable

import pandas as pd


class BaseExtractor:
    schema: dict[str, Any] = {}
    extra_columns_schema: dict[str, Any] = {"message_id": pd.Int32Dtype()}
    field: str = ""

    @classmethod
    def _extra_columns(
        cls,
        _message: dict[str, Any],
        _measurements: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        return {}

    @classmethod
    def extract(cls, messages: Iterable[dict[str, Any]]) -> pd.DataFrame:
        data = {col: [] for col in cls.schema | cls.extra_columns_schema}

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
                for col, dtype in (cls.schema | cls.extra_columns_schema).items()
            }
        )
