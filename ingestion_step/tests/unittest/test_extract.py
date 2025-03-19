# pyright: reportPrivateUsage=false
from typing import Any

import ingestion_step.ztf.extractor as extractor


def test_extract_candidates(ztf_alerts: list[dict[str, Any]]):
    candidates = extractor._extract_candidates(ztf_alerts)
    assert len(ztf_alerts) == len(candidates)

    for candidate in candidates:
        assert "objectId" in candidate
        assert "candid" in candidate
        assert "parent_candid" in candidate
        assert "has_stamp" in candidate


def test_extract_prv_candidates(ztf_alerts: list[dict[str, Any]]):
    prv_candidates = extractor._extract_prv_candidates(ztf_alerts)

    for prv_candidate in prv_candidates:
        assert "objectId" in prv_candidate
        assert "parent_candid" in prv_candidate
        assert "has_stamp" in prv_candidate


def test_extract_fp_hist(ztf_alerts: list[dict[str, Any]]):
    fp_hists = extractor._extract_fp_hist(ztf_alerts)

    for fp_hist in fp_hists:
        assert "objectId" in fp_hist
        assert "ra" in fp_hist
        assert "dec" in fp_hist
        assert "magzpsci" in fp_hist
        assert "forcediffimflux" in fp_hist
        assert "forcediffimfluxunc" in fp_hist
        assert "parent_candid" in fp_hist
        assert "has_stamp" in fp_hist


def test_extract(ztf_alerts: list[dict[str, Any]]):
    ztf_data = extractor.extract(ztf_alerts)

    cands = ztf_data["candidates"]
    prv_cands = ztf_data["prv_candidates"]
    fp_hists = ztf_data["fp_hist"]

    assert len(ztf_alerts) == len(cands)

    raise NotImplementedError
