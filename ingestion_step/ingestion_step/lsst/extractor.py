from typing import Any, TypedDict

import pandas as pd

from ingestion_step.core.extractor_interface import BaseExtractor

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


class LsstSourceExtractor(BaseExtractor):
    field = "diaSource"
    schema = dia_source_schema
    extra_columns_schema = BaseExtractor.extra_columns_schema | {
        "has_stamp": pd.BooleanDtype()
    }

    @staticmethod
    def _has_stamp(message: dict[str, Any]) -> bool:
        return (
            message["cutoutDifference"] is not None
            and message["cutoutScience"] is not None
            and message["cutoutTemplate"] is not None
        )

    @classmethod
    def _extra_columns(
        cls,
        message: dict[str, Any],
        measurements: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        return {"has_stamp": [cls._has_stamp(message)]}


class LsstPrvSourceExtractor(BaseExtractor):
    field = "prvDiaSources"
    schema = dia_source_schema
    extra_columns_schema = BaseExtractor.extra_columns_schema | {
        "has_stamp": pd.BooleanDtype()
    }

    @classmethod
    def _extra_columns(
        cls,
        message: dict[str, Any],
        measurements: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        return {"has_stamp": [False] * len(measurements)}


class LsstForcedSourceExtractor(BaseExtractor):
    field = "prvDiaForcedSources"
    schema = dia_forced_source_schema


class LsstNonDetectionsExtractor(BaseExtractor):
    field = "prvDiaNondetectionLimits"
    schema = dia_non_detection_limit_schema


class LsstDiaObjectExtractor(BaseExtractor):
    field = "diaObject"
    schema = dia_object_schema


class LsstSsObjectExtractor(BaseExtractor):
    field = "ssObject"
    schema = ss_object_schema


def extract(messages: list[dict[str, Any]]):
    """
    Returns the `LSSTData` of the batch of messages.

    Extracts from each message it's 'sources', 'previous_sources', 'forced_sources'
    and 'non_detections'.
    Adds to each necessary fields from the alert itself (so some data is
    duplicated between dataframes)
    """
    return LSSTData(
        sources=LsstSourceExtractor.extract(messages),
        previous_sources=LsstPrvSourceExtractor.extract(messages),
        forced_sources=LsstForcedSourceExtractor.extract(messages),
        non_detections=LsstNonDetectionsExtractor.extract(messages),
        dia_object=LsstDiaObjectExtractor.extract(messages),
        ss_object=LsstSsObjectExtractor.extract(messages),
    )
