from typing import Any

import pandas as pd

from ingestion_step.core.extractor import BaseExtractor
from ingestion_step.core.types import Message

from .schemas import (
    dia_forced_source_schema,
    # dia_non_detection_limit_schema,
    dia_object_schema,
    dia_source_schema,
    mpcorb_schema,
    ss_source_schema,
    # ss_object_schema,
)


class LsstDiaSourceExtractor(BaseExtractor):
    field = "diaSource"
    schema = dia_source_schema
    extra_columns_schema = {"has_stamp": pd.BooleanDtype()}

    @staticmethod
    def _has_stamp(message: Message) -> bool:
        return (
            message["cutoutDifference"] is not None
            and message["cutoutScience"] is not None
            and message["cutoutTemplate"] is not None
        )

    @classmethod
    def _extra_columns(
        cls,
        message: Message,
        measurements: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        return {"has_stamp": [cls._has_stamp(message)]}


class LsstSsSourceExtractor(BaseExtractor):
    field = "ssSource"
    schema = ss_source_schema


class LsstPrvSourceExtractor(BaseExtractor):
    field = "prvDiaSources"
    schema = dia_source_schema
    extra_columns_schema = {"has_stamp": pd.BooleanDtype()}

    @classmethod
    def _extra_columns(
        cls,
        message: Message,
        measurements: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        return {"has_stamp": [False] * len(measurements)}


class LsstForcedSourceExtractor(BaseExtractor):
    field = "prvDiaForcedSources"
    schema = dia_forced_source_schema


# class LsstNonDetectionsExtractor(BaseExtractor):
#     field = "prvDiaNondetectionLimits"
#     schema = dia_non_detection_limit_schema
#     extra_columns_schema = {
#         "diaObjectId": pd.Int64Dtype(),
#         "ssObjectId": pd.Int64Dtype(),
#     }
#
#     @classmethod
#     def _extra_columns(
#         cls, message: Message, measurements: list[dict[str, Any]]
#     ) -> dict[str, list[Any]]:
#         source = message["diaSource"]
#         return {
#             "diaObjectId": [source["diaObjectId"]] * len(measurements),
#             "ssObjectId": [source["ssObjectId"]] * len(measurements),
#         }


class LsstDiaObjectExtractor(BaseExtractor):
    field = "diaObject"
    schema = dia_object_schema
    extra_columns_schema = {
        "ra": pd.Float64Dtype(),
        "dec": pd.Float64Dtype(),
        "midpointMjdTai": pd.Float64Dtype(),
    }

    @classmethod
    def _extra_columns(
        cls, message: Message, measurements: list[dict[str, Any]]
    ) -> dict[str, list[Any]]:
        source = message["diaSource"]
        return {
            "midpointMjdTai": [source["midpointMjdTai"]] * len(measurements),
        }


# class LsstSsObjectExtractor(BaseExtractor):
#     field = "ssObject"
#     schema = ss_object_schema
#     extra_columns_schema = {
#         "ra": pd.Float64Dtype(),
#         "dec": pd.Float64Dtype(),
#         "midpointMjdTai": pd.Float64Dtype(),
#     }
#
#     @classmethod
#     def _extra_columns(
#         cls, message: Message, measurements: list[dict[str, Any]]
#     ) -> dict[str, list[Any]]:
#         source = message["diaSource"]
#         return {
#             "ra": [source["ra"]] * len(measurements),
#             "dec": [source["dec"]] * len(measurements),
#             "midpointMjdTai": [source["midpointMjdTai"]] * len(measurements),
#         }


class LsstMpcorbExtractor(BaseExtractor):
    field = "MPCORB"
    schema = mpcorb_schema
