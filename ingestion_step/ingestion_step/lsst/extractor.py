from typing import Any, TypedDict

import pandas as pd

from .schemas import (
    dia_forced_source_schema,
    dia_non_detection_limit_schema,
    dia_object_schema,
    dia_source_schema,
    ss_object_schema,
)


class LSSTData(TypedDict):
    """
    Dictionary of pandas DataFrames containing the different kinds of data inside
    the messages sent by LSST.
    """

    sources: pd.DataFrame
    previous_sources: pd.DataFrame
    forced_sources: pd.DataFrame
    non_detections: pd.DataFrame
    dia_object: pd.DataFrame
    ss_object: pd.DataFrame


def _extract_sources_from_field(
    messages: list[dict[str, Any]],
    field: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract all sources for the list of messages.

    Returns a dictionary of list with the content of `field` of each
    alert and add some extra necessary fields from the alert to each one.
    """
    source_schema = schema | {
        "message_id": pd.Int32Dtype(),
        "alertId": pd.Int64Dtype(),
    }
    sources = {col: [] for col in source_schema}

    for message_id, message in enumerate(messages):
        if message[field] is None:
            message[field] = []
        if type(message[field]) is not list:
            message[field] = [message[field]]
        for source in message[field]:
            for col, value in source.items():
                sources[col].append(value)
            sources["message_id"].append(message_id)
            sources["alertId"].append(message["alertId"])

    return {
        col: pd.Series(sources[col], dtype=dtype)
        for col, dtype in source_schema.items()
    }


def extract(messages: list[dict[str, Any]]):
    """
    Returns the `LSSTData` of the batch of messages.

    Extracts from each message it's 'sources', 'previous_sources', 'forced_sources'
    and 'non_detections'.
    Adds to each necessary fields from the alert itself (so some data is
    duplicated between dataframes)
    """
    return LSSTData(
        sources=pd.DataFrame(
            _extract_sources_from_field(
                messages, "diaSource", dia_source_schema
            )
        ),
        previous_sources=pd.DataFrame(
            _extract_sources_from_field(
                messages, "prvDiaSources", dia_source_schema
            )
        ),
        forced_sources=pd.DataFrame(
            _extract_sources_from_field(
                messages, "prvDiaForcedSources", dia_forced_source_schema
            )
        ),
        non_detections=pd.DataFrame(
            _extract_sources_from_field(
                messages,
                "prvDiaNondetectionLimits",
                dia_non_detection_limit_schema,
            )
        ),
        dia_object=pd.DataFrame(
            _extract_sources_from_field(
                messages, "diaObject", dia_object_schema
            )
        ),
        ss_object=pd.DataFrame(
            _extract_sources_from_field(messages, "ssObject", ss_object_schema)
        ),
    )
