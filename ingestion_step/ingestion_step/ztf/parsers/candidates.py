from typing import NamedTuple

import numpy as np
import pandas as pd

from ingestion_step.ztf.parsers.transforms import (
    add_sid,
    add_tid,
    apply_transforms,
    candid_to_measurement_id,
    fid_to_band,
    isdiffpos_to_int,
    jd_to_mjd,
    magpsf_to_mag,
    objectId_to_oid,
    sigmadec_to_e_dec,
    sigmapsf_to_e_mag,
    sigmara_to_e_ra,
)

CANDIDATES_TRANSFORMS = [
    objectId_to_oid,
    candid_to_measurement_id,
    add_tid,
    add_sid,
    isdiffpos_to_int,
    fid_to_band,
    jd_to_mjd,
    sigmara_to_e_ra,
    sigmadec_to_e_dec,
    magpsf_to_mag,
    sigmapsf_to_e_mag,
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
    objects = candidates.replace({np.nan: None})

    return objects


def _parse_dets_from_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    detections = candidates.replace({np.nan: None})

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
