from typing import NamedTuple

import numpy as np
import pandas as pd

from ingestion_step.ztf.parsers.transforms import (
    add_drb,
    add_drbversion,
    add_rfid,
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

PRV_CANDIDATES_TRANSFORMS = [
    objectId_to_oid,
    candid_to_measurement_id,
    add_tid,
    add_sid,
    sigmara_to_e_ra,
    sigmadec_to_e_dec,
    fid_to_band,
    jd_to_mjd,
    magpsf_to_mag,
    sigmapsf_to_e_mag,
    isdiffpos_to_int,
    add_drb,
    add_drbversion,
    add_rfid,
]
"""
List of mappings applied to the 'prv_candidates' `DataFrame` before extracting each
subset of columns.
"""


class ParsedPrvCandidates(NamedTuple):
    detections: pd.DataFrame
    non_detections: pd.DataFrame


def _parse_dets_from_prv_candidates(
    prv_candidates: pd.DataFrame,
) -> pd.DataFrame:
    det_prv_candidates = prv_candidates[prv_candidates["candid"].notnull()]

    return det_prv_candidates


def _parse_non_dets_from_prv_candidates(
    prv_candidates: pd.DataFrame,
) -> pd.DataFrame:
    non_det_prv_candidates = prv_candidates[prv_candidates["candid"].isnull()]

    return non_det_prv_candidates


def parse_prv_candidates(
    prv_candidates: pd.DataFrame,
) -> ParsedPrvCandidates:
    """
    Parses a `DataFrame` of prv_candidates into `detections` and `non_detections`.

    Apply a series of mappings to the original `DataFrame` *mutating it in
    place* then extracts a subset of it`s columns to form the new `DataFrames`.
    """
    apply_transforms(prv_candidates, PRV_CANDIDATES_TRANSFORMS)

    detections = _parse_dets_from_prv_candidates(prv_candidates)
    non_detections = _parse_non_dets_from_prv_candidates(prv_candidates)

    return ParsedPrvCandidates(
        detections=detections, non_detections=non_detections
    )
