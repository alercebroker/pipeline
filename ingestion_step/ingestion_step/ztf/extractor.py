from typing import Any, TypedDict

import pandas as pd


class ZTFData(TypedDict):
    """
    Dictionary of pandas DataFrames containing the different kinds of data inside
    the messages sent by ZTF.
    """

    candidates: pd.DataFrame
    prv_candidates: pd.DataFrame
    fp_hists: pd.DataFrame


def _has_stamp(message: dict[str, Any]) -> bool:
    """
    Returns `True` if the given message contains all the stamp related fields.
    """
    return (
        ("cutoutScience" in message and message["cutoutScience"] is not None)
        and ("cutoutTemplate" in message and message["cutoutTemplate"] is not None)
        and ("cutoutDifference" in message and message["cutoutDifference"] is not None)
    )


def _extract_candidates(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract all candidates for the list of messages.

    Returns a list with the content of 'candidate' of each alert and add some
    extra necessary fields from the alert to each one.
    """
    return [
        {
            "message_id": message_id,
            "objectId": message["objectId"],
            "candid": message["candid"],
            "parent_candid": None,
            "has_stamp": _has_stamp(message),
            "forced": False,
            **message["candidate"],
        }
        for message_id, message in enumerate(messages)
    ]


def _extract_prv_candidates(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract all previous candidates for the list of messages.

    Returns a flat list with the content of 'prv_candidates' of each alert where it
    exists and add some extra necessary fields from the alert to each one.
    """
    prv_candidates = []
    for message_id, message in enumerate(messages):
        if "prv_candidates" not in message or message["prv_candidates"] is None:
            continue
        for prv_candidate in message["prv_candidates"]:
            prv_candidates.append(
                {
                    "message_id": message_id,
                    "objectId": message["objectId"],
                    "parent_candid": message["candid"],
                    "has_stamp": _has_stamp(message),
                    "forced": False,
                    **prv_candidate,
                }
            )

    return prv_candidates


def _extract_fp_hists(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract all forced photometries for the list of messages.

    Returns a flat list with the content of 'fp_hists' of each alert where it
    exists and add some extra necessary fields from the alert to each one.
    """

    fp_hists = []
    for message_id, message in enumerate(messages):
        if "fp_hists" not in message or message["fp_hists"] is None:
            continue
        for fp_hist in message["fp_hists"]:
            fp_hists.append(
                {
                    "message_id": message_id,
                    "objectId": message["objectId"],
                    "ra": message["candidate"]["ra"],
                    "dec": message["candidate"]["dec"],
                    "magzpsci": message["candidate"]["magzpsci"],
                    "parent_candid": message["candid"],
                    "has_stamp": _has_stamp(message),
                    "forced": True,
                    **fp_hist,
                }
            )

    return fp_hists


def extract(messages: list[dict[str, Any]]):
    """
    Returns the `ZTFData` of the batch of messages.

    Extracts from each message it's 'candidates', 'prv_candidates' and 'fp_hists'
    and adds to each neccesary fields from the alert itself (so some data is
    duplicated between dataframes)

    'prv_candidates' and 'fp_hists' are flattened and *may be empty*, as
    zero or more of them can be in one alert.
    """
    return ZTFData(
        candidates=pd.DataFrame(_extract_candidates(messages)),
        prv_candidates=pd.DataFrame(_extract_prv_candidates(messages)),
        fp_hists=pd.DataFrame(_extract_fp_hists(messages)),
    )
