from typing import Any

import pandas as pd

from ingestion_step.core.extractor import BaseExtractor
from ingestion_step.core.types import Message
from ingestion_step.ztf.schemas import (
    candidate_schema,
    fp_hist_schema,
    prv_candidate_schema,
)


class ZtfCandidatesExtractor(BaseExtractor):
    field = "candidate"
    schema = candidate_schema
    extra_columns_schema = {
        "objectId": pd.StringDtype(),
        "parent_candid": pd.Int64Dtype(),
        "has_stamp": pd.BooleanDtype(),
        "forced": pd.BooleanDtype(),
    }

    @staticmethod
    def _has_stamp(message: Message) -> bool:
        return (
            (
                "cutoutScience" in message
                and message["cutoutScience"] is not None
            )
            and (
                "cutoutTemplate" in message
                and message["cutoutTemplate"] is not None
            )
            and (
                "cutoutDifference" in message
                and message["cutoutDifference"] is not None
            )
        )

    @classmethod
    def _extra_columns(
        cls,
        message: Message,
        measurements: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        return {
            "objectId": [message["objectId"]],
            "parent_candid": [None],
            "has_stamp": [cls._has_stamp(message)],
            "forced": [False],
        }


class ZtfPrvCandidatesExtractor(BaseExtractor):
    field = "prv_candidates"
    schema = prv_candidate_schema
    extra_columns_schema = {
        "objectId": pd.StringDtype(),
        "parent_candid": pd.Int64Dtype(),
        "has_stamp": pd.BooleanDtype(),
        "forced": pd.BooleanDtype(),
    }

    @classmethod
    def _extra_columns(
        cls,
        message: Message,
        measurements: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        return {
            "objectId": [message["objectId"]] * len(measurements),
            "parent_candid": [message["candid"]] * len(measurements),
            "has_stamp": [False] * len(measurements),
            "forced": [False] * len(measurements),
        }


class ZtfFpHistsExtractor(BaseExtractor):
    field = "fp_hists"
    schema = fp_hist_schema
    extra_columns_schema = {
        "objectId": pd.StringDtype(),
        "parent_candid": pd.Int64Dtype(),
        "has_stamp": pd.BooleanDtype(),
        "forced": pd.BooleanDtype(),
        "ra": pd.Float64Dtype(),
        "dec": pd.Float64Dtype(),
    }

    @classmethod
    def _extra_columns(
        cls,
        message: Message,
        measurements: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        return {
            "objectId": [message["objectId"]] * len(measurements),
            "parent_candid": [message["candid"]] * len(measurements),
            "has_stamp": [False] * len(measurements),
            "forced": [True] * len(measurements),
            "ra": [message["candidate"]["ra"]] * len(measurements),
            "dec": [message["candidate"]["dec"]] * len(measurements),
        }
