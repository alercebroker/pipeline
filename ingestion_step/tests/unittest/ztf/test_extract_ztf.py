from ingestion_step.core.types import Message
from ingestion_step.ztf.extractor import (
    ZtfCandidatesExtractor,
    ZtfFpHistsExtractor,
    ZtfPrvCandidatesExtractor,
)


def test_extract_candidates(ztf_alerts: list[Message]):
    candidates = ZtfCandidatesExtractor.extract(ztf_alerts)
    assert len(candidates) == len(ztf_alerts)

    fields = {"objectId", "candid", "parent_candid", "has_stamp"}
    assert fields <= set(candidates.keys())
    assert not any(candidates["forced"]), "`forced` should be false on candidates"


def test_extract_prv_candidates(ztf_alerts: list[Message]):
    prv_candidates = ZtfPrvCandidatesExtractor.extract(ztf_alerts)

    fields = {"objectId", "parent_candid", "has_stamp"}
    assert fields <= set(prv_candidates.keys())
    assert not any(
        prv_candidates["forced"]
    ), "`forced` should be false on prv_candidates"


def test_extract_fp_hists(ztf_alerts: list[Message]):
    fp_hists = ZtfFpHistsExtractor.extract(ztf_alerts)

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

    assert fields <= set(fp_hists.keys())
    assert all(fp_hists["forced"]), "`forced` should be true on fp"
