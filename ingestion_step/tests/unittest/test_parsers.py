# pyright: reportPrivateUsage=false
from typing import Callable

import pandas as pd
import pytest

from ingestion_step.ztf.extractor import ZTFData
from ingestion_step.ztf.parsers import prv_candidates


def apply_transforms(
    ztf_data: ZTFData,
    df: str,
    transforms: list[Callable[[pd.DataFrame], None]],
):
    for transform in transforms:
        try:
            transform(ztf_data[df])
        except Exception as exception:
            pytest.fail(
                f"Could not apply `{transform.__name__}` to `{df}`. \n{exception=}"
            )


def test_prv_candidates_parser(ztf_data: ZTFData):
    apply_transforms(
        ztf_data, "prv_candidates", prv_candidates.PRV_CANDIDATES_TRANSFORMS
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
