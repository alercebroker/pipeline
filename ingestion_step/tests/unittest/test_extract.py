# pyright: reportPrivateUsage=false
from typing import Any

import ingestion_step.ztf.extractor as extractor


def test_extract_candidates(ztf_alerts: list[dict[str, Any]]):
    candidates = extractor._extract_candidates(ztf_alerts)
    assert len(ztf_alerts) == len(candidates)

    fields = {"objectId", "candid", "parent_candid", "has_stamp"}
    for candidate in candidates:
        assert fields <= set(candidate.keys())


def test_extract_prv_candidates(ztf_alerts: list[dict[str, Any]]):
    prv_candidates = extractor._extract_prv_candidates(ztf_alerts)

    fields = {"objectId", "parent_candid", "has_stamp"}
    for prv_candidate in prv_candidates:
        assert fields <= set(prv_candidate.keys())


def test_extract_fp_hists(ztf_alerts: list[dict[str, Any]]):
    fp_hists = extractor._extract_fp_hists(ztf_alerts)

    fields = {
        "objectId",
        "ra",
        "dec",
        "magzpsci",
        "forcediffimflux",
        "forcediffimfluxunc",
        "parent_candid",
        "has_stamp",
    }
    for fp_hist in fp_hists:
        assert fields <= set(fp_hist.keys())


def test_extract(ztf_alerts: list[dict[str, Any]]):
    ztf_data = extractor.extract(ztf_alerts)

    n_candidates = len(ztf_alerts)
    n_prv_candidates = sum(
        len(alert["prv_candidates"])
        for alert in ztf_alerts
        if alert["prv_candidates"]
    )
    n_fp_hists = sum(
        len(alert["fp_hists"]) for alert in ztf_alerts if alert["fp_hists"]
    )

    assert n_candidates == len(ztf_data["candidates"])
    assert n_prv_candidates == len(ztf_data["prv_candidates"])
    assert n_fp_hists == len(ztf_data["fp_hists"])
