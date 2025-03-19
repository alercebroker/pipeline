from typing import NamedTuple

import pandas as pd

from ingestion_step.ztf.parsers.transforms import (
    add_sid,
    add_tid,
    apply_transforms,
    candid_to_measurment_id,
    fid_to_band,
    jd_to_mjd,
    objectId_to_oid,
)

candidates_tansforms = [
    objectId_to_oid,
    candid_to_measurment_id,
    add_tid,
    add_sid,
    fid_to_band,
    jd_to_mjd,
]


class ParsedCandidates(NamedTuple):
    objects: pd.DataFrame
    detections: pd.DataFrame


def _parse_objs_from_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    cols = ["oid", "sid", "tid"]

    objects = candidates[cols]

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

    detections = candidates[cols]

    return detections


def parse_candidates(
    candidates: pd.DataFrame,
) -> ParsedCandidates:
    apply_transforms(candidates, candidates_tansforms)

    objects = _parse_objs_from_candidates(candidates)
    detections = _parse_dets_from_candidates(candidates)

    return ParsedCandidates(
        objects=objects,
        detections=detections,
    )
