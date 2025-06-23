from typing import Any, TypedDict

import pandas as pd

from .schemas.candidate import candidate_schema
from .schemas.fp_hist import fp_hist_schema
from .schemas.prv_candidate import prv_candidate_schema


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
        and (
            "cutoutTemplate" in message
            and message["cutoutTemplate"] is not None
        )
        and (
            "cutoutDifference" in message
            and message["cutoutDifference"] is not None
        )
    )


def _extract_candidates(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Extract all candidates for the list of messages.

    Returns a dictionary of list with the content of 'candidate' of each
    alert and add some extra necessary fields from the alert to each one.
    """
    candidates = {col: [] for col in candidate_schema}

    for message_id, message in enumerate(messages):
        for col, value in message["candidate"].items():
            if col in ["objectidps1", "objectidps2", "objectidps3", "tblid"]:
                if value is not None:
                    candidates[col].append(str(value))
                else:
                    candidates[col].append(None)
            else:
                candidates[col].append(value)

        candidates["message_id"].append(message_id)
        candidates["objectId"].append(message["objectId"])
        candidates["parent_candid"].append(None)
        candidates["has_stamp"].append(_has_stamp(message))
        candidates["forced"].append(False)

    return {
        col: pd.Series(candidates[col], dtype=dtype())
        for col, dtype in candidate_schema.items()
    }


def _extract_prv_candidates(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Extract all previous candidates for the list of messages.

    Returns a dictionary of list with the content of 'prv_candidates' of
    each alert where it exists and add some extra necessary fields from
    the alert to each one.
    """
    prv_candidates = {col: [] for col in prv_candidate_schema}

    for message_id, message in enumerate(messages):
        if message["prv_candidates"] is None:
            continue
        for prv_candidate in message["prv_candidates"]:
            for col, value in prv_candidate.items():
                if col in ["tblid"]:
                    if value is not None:
                        prv_candidates[col].append(str(value))
                    else:
                        prv_candidates[col].append(None)
                else:
                    prv_candidates[col].append(value)
            prv_candidates["message_id"].append(message_id)
            prv_candidates["objectId"].append(message["objectId"])
            prv_candidates["parent_candid"].append(message["candid"])
            prv_candidates["has_stamp"].append(False)
            prv_candidates["forced"].append(False)

    return {
        col: pd.Series(prv_candidates[col], dtype=dtype())
        for col, dtype in prv_candidate_schema.items()
    }


def _extract_fp_hists(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Extract all forced photometries for the list of messages.

    Returns a dictionary of lists with the content of 'fp_hists' of each
    alert where it exists and add some extra necessary fields from the
    alert to each one.
    """
    fp_hists = {col: [] for col in fp_hist_schema}

    for message_id, message in enumerate(messages):
        if message["fp_hists"] is None:
            continue
        for fp_hist in message["fp_hists"]:
            for col, value in fp_hist.items():
                fp_hists[col].append(value)
            fp_hists["ra"].append(message["candidate"]["ra"])
            fp_hists["dec"].append(message["candidate"]["dec"])
            fp_hists["message_id"].append(message_id)
            fp_hists["objectId"].append(message["objectId"])
            fp_hists["parent_candid"].append(message["candid"])
            fp_hists["has_stamp"].append(False)
            fp_hists["forced"].append(True)

    return {
        col: pd.Series(fp_hists[col], dtype=dtype())
        for col, dtype in fp_hist_schema.items()
    }


def extract(messages: list[dict[str, Any]]):
    """
    Returns the `ZTFData` of the batch of messages.

    Extracts from each message it's 'candidates', 'prv_candidates' and 'fp_hists'
    and adds to each necessary fields from the alert itself (so some data is
    duplicated between dataframes)

    'prv_candidates' and 'fp_hists' are flattened and *may be empty*, as
    zero or more of them can be in one alert.
    """
    return ZTFData(
        candidates=pd.DataFrame(_extract_candidates(messages)),
        prv_candidates=pd.DataFrame(_extract_prv_candidates(messages)),
        fp_hists=pd.DataFrame(_extract_fp_hists(messages)),
    )
