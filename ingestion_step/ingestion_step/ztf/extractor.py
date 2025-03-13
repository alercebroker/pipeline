from typing import Any, TypedDict

import pandas as pd


class ZTFData(TypedDict):
    candidates: pd.DataFrame
    prv_candidates: pd.DataFrame
    fp_hist: pd.DataFrame


def _has_stamp(message: dict[str, Any]) -> bool:
    return (
        "cutoutScience" in message
        and "cutoutTemplate" in message
        and "cutoutDifference" in message
    )


def _extract_candidates(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "objectId": message["objectId"],
            "candid": message["candid"],
            "parent_candid": None,
            "has_stamp": _has_stamp(message),
            **message["candidate"],
        }
        for message in messages
    ]


def _extract_prv_candidates(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "objectId": message["objectId"],
            "parent_candid": message["candid"],
            "has_stamp": _has_stamp(message),
            **prv_candidate,
        }
        for message in messages
        for prv_candidate in message["prv_candidates"]
    ]


def _extract_fp_hist(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "objectId": message["objectId"],
            "ra": message["candidate"]["ra"],
            "dec": message["candidate"]["dec"],
            "magzpsci": message["candidate"]["magzpsci"],
            "forcediffimflux": message["candidate"]["forcediffimflux"],
            "forcediffimfluxunc": message["candidate"]["forcediffimfluxunc"],
            "parent_candid": message["candid"],
            "has_stamp": _has_stamp(message),
            **fp_hist,
        }
        for message in messages
        for fp_hist in message["fp_hists"]
    ]


def extract(messages: list[dict[str, Any]]):
    return ZTFData(
        candidates=pd.DataFrame(_extract_candidates(messages)),
        prv_candidates=pd.DataFrame(_extract_prv_candidates(messages)),
        fp_hist=pd.DataFrame(_extract_fp_hist(messages)),
    )
