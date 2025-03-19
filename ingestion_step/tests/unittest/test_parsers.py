# pyright: reportPrivateUsage=false
from typing import Callable

import pandas as pd
import pytest

from ingestion_step.ztf.extractor import ZTFData
from ingestion_step.ztf.parsers import candidates, fp_hist, prv_candidates


def apply_transforms(
    ztf_data: ZTFData, df: str, transforms: list[Callable[[pd.DataFrame], None]]
):
    for transform in transforms:
        try:
            transform(ztf_data[df])
        except Exception as exception:
            pytest.fail(
                f"Could not apply `{transform.__name__}` to `{df}`. \n{exception=}"
            )


def test_parse_candidates(ztf_data: ZTFData):
    apply_transforms(ztf_data, "candidates", candidates.candidates_tansforms)

    try:
        candidates._parse_objs_from_candidates(ztf_data["candidates"])
    except Exception as exception:
        pytest.fail(f"Could not parse `objects` from `candidates`.\n{exception=}")

    try:
        candidates._parse_dets_from_candidates(ztf_data["candidates"])
    except Exception as exception:
        pytest.fail(f"Could not parse `objects` from `candidates`.\n{exception=}")


def test_prv_candidates_parser(ztf_data: ZTFData):
    apply_transforms(
        ztf_data, "prv_candidates", prv_candidates.prv_candidates_tansforms
    )

    try:
        prv_candidates._parse_dets_from_prv_candidates(ztf_data["prv_candidates"])
    except Exception as exception:
        pytest.fail(
            f"Could not parse `detections` from `prv_candidates`.\n{exception=}"
        )

    try:
        prv_candidates._parse_non_dets_from_prv_candidates(ztf_data["prv_candidates"])
    except Exception as exception:
        pytest.fail(
            f"Could not parse `non_detections` from `prv_candidates`.\n{exception=}"
        )


def test_fp_hist_parser(ztf_data: ZTFData):
    apply_transforms(ztf_data, "fp_hist", fp_hist.fp_transforms)

    try:
        fp_hist._parse_fps_from_fp_hist(ztf_data["fp_hist"])
    except Exception as exception:
        pytest.fail(
            f"Could not parse `forced_photometries` from `fp_hist`.\n{exception=}"
        )


def test_ztf_parser(ztf_data: ZTFData):
    raise NotImplementedError
