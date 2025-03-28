from typing import NamedTuple

import numpy as np
import pandas as pd

from ingestion_step.ztf.parsers.transforms import (
    add_sid,
    add_tid,
    apply_transforms,
    candid_to_measurment_id,
    fid_to_band,
    isdiffpos_to_int,
    jd_to_mjd,
    objectId_to_oid,
)

CANDIDATES_TRANSFORMS = [
    objectId_to_oid,
    candid_to_measurment_id,
    add_tid,
    add_sid,
    isdiffpos_to_int,
    fid_to_band,
    jd_to_mjd,
]
"""
List of mappings applied to the 'candidates' `DataFrame` before extracting each
subset of columns.
"""


class ParsedCandidates(NamedTuple):
    objects: pd.DataFrame
    detections: pd.DataFrame


def _parse_objs_from_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    cols = ["oid", "sid", "tid", "ra", "dec", "mjd"]

    objects = candidates[cols].replace({np.nan: None})

    return objects


def _parse_dets_from_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "oid",
        "measurement_id",
        "ra",
        "dec",
        "band",
        "mjd",
        "pid",
        "diffmaglim",
        "isdiffpos",
        "nid",
        "magpsf",
        "sigmapsf",
        "magap",
        "sigmagap",
        "distnr",
        "rb",
        "rbversion",
        "drb",
        "drbversion",
        "magapbig",
        "sigmagapbig",
        "parent_candid",
        "rfid",
        # "magpsf_corr",
        # "sigmapsf_corr",
        # "sigmapsf_corr_ext",
        # "corrected",
        # "dubious",
        "has_stamp",
        # "step_id_corr",
    ]

    detections = candidates[cols].replace({np.nan: None})

    return detections


def parse_candidates(
    candidates: pd.DataFrame,
) -> ParsedCandidates:
    """
    Parses a `DataFrame` of candidates into `objects` and `detections`.

    Apply a series of mappings to the original `DataFrame` *mutating it in
    place* then extracts a subset of it`s columns to form the new `DataFrames`.
    """
    apply_transforms(candidates, CANDIDATES_TRANSFORMS)

    objects = _parse_objs_from_candidates(candidates)
    detections = _parse_dets_from_candidates(candidates)

    return ParsedCandidates(
        objects=objects,
        detections=detections,
    )
