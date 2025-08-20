from typing import Any, Callable

import pandas as pd

from ingestion_step.core.types import DType

Transform = Callable[[pd.DataFrame], None]


def apply_transforms(df: pd.DataFrame, transforms: list[Transform]):
    """Applies a list of transforms to the given dataframe"""
    for transform in transforms:
        transform(df)


def copy_column(src: str, dest: str):
    def _copy_column(df: pd.DataFrame):
        df[dest] = df[src]

    return _copy_column


def rename_column(src: str, dest: str):
    def _rename_column(df: pd.DataFrame):
        df.rename(columns={src: dest}, inplace=True)

    return _rename_column


def add_constant_column(column: str, constant: Any, dtype: DType):
    def _add_constant_column(df: pd.DataFrame):
        df[column] = pd.Series([constant] * len(df), dtype=dtype)

    return _add_constant_column


def groupby_messageid(df: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    return (
        df.groupby("message_id")
        .apply(lambda x: x.to_dict("records"), include_groups=False)
        .to_dict()
    )
