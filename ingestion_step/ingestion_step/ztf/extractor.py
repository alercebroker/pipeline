from typing import Any, TypedDict

import pandas as pd


class ZTFData(TypedDict):
    candidates: pd.DataFrame
    prv_candidates: pd.DataFrame
    fp_hists: pd.DataFrame


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
    prv_candidates = []
    for message in messages:
        if "prv_candidates" not in message or message["prv_candidates"] is None:
            continue
        for prv_candidate in message["prv_candidates"]:
            prv_candidates.append(
                {
                    "objectId": message["objectId"],
                    "parent_candid": message["candid"],
                    "has_stamp": _has_stamp(message),
                    **prv_candidate,
                }
            )

    return prv_candidates


def _extract_fp_hists(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fp_hists = []
    for message in messages:
        if "fp_hists" not in message or message["fp_hists"] is None:
            continue
        for fp_hist in message["fp_hists"]:
            fp_hists.append(
                {
                    "objectId": message["objectId"],
                    "ra": message["candidate"]["ra"],
                    "dec": message["candidate"]["dec"],
                    "magzpsci": message["candidate"]["magzpsci"],
                    "parent_candid": None,
                    "has_stamp": _has_stamp(message),
                    **fp_hist,
                }
            )

    return fp_hists


def extract(messages: list[dict[str, Any]]):
    return ZTFData(
        candidates=pd.DataFrame(_extract_candidates(messages)),
        prv_candidates=pd.DataFrame(_extract_prv_candidates(messages)),
        fp_hists=pd.DataFrame(_extract_fp_hists(messages)),
    )
