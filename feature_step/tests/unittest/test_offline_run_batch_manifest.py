"""The unit manifest must record how many oids got no AllWISE counterpart.

Without this number, "Xwave returned nothing for this object" and "the
crossmatch never ran for this object" are indistinguishable in everything the
run leaves behind: both end up with no WISE rows in <schema>.feature and no row
in <schema>.xmatch. Diagnosing 27.5.7a32.dev1 cost a full model ablation for
exactly that reason. The expected rate is ~14% (WISE_NULL_CLASSIFICATION_IMPACT.md
puts recovery at 86%), so a run that reports 40% is telling you something is
wrong with the crossmatch, not with the sky.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PIPE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPE / "feature_step" / "scripts"))

import offline_run_batch as R


def _inputs(oids, with_allwise):
    """{oid: (message, refs, allwise)} — allwise empty for the oids without one."""
    cols = ["oid", "W1", "W2", "W3", "W4"]
    out = {}
    for oid in oids:
        allwise = (pd.DataFrame([{"oid": oid, "W1": 1.0, "W2": 2.0, "W3": 3.0, "W4": 4.0}])
                   if oid in with_allwise else pd.DataFrame(columns=cols))
        out[oid] = ({"oid": oid}, pd.DataFrame(), allwise)
    return out


def test_manifest_counts_oids_with_no_allwise_match(monkeypatch, tmp_path):
    oids = [1, 2, 3, 4]
    monkeypatch.setattr(R, "_W", {"cfg": {
        "out_dir": str(tmp_path), "minibatch": 500, "features": False,
        "credentials": "creds", "schema": "multisurvey_ztf", "retries": 0,
        "xmatch_url": "http://x", "min_detections": 1,
    }})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise={1, 3}))
    monkeypatch.setattr(R, "process_oid",
                        lambda oid, *a, **k: ([{"oid": oid}], None))

    manifest = R.process_unit((0, oids))

    assert manifest["n_ok"] == 4              # all four classified fine...
    assert manifest["n_no_allwise"] == 2      # ...but two had no counterpart


def test_oids_without_detections_are_not_counted_as_missing_allwise(monkeypatch, tmp_path):
    """No detections means no crossmatch was ever attempted, so it is not a miss.

    Counting them would inflate the rate with objects the crossmatch never saw.
    """
    monkeypatch.setattr(R, "_W", {"cfg": {
        "out_dir": str(tmp_path), "minibatch": 500, "features": False,
        "credentials": "creds", "schema": "multisurvey_ztf", "retries": 0,
        "xmatch_url": "http://x", "min_detections": 1,
    }})
    # oid 2 is absent from the returned inputs: it has no detections.
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs([1], with_allwise=set()))
    monkeypatch.setattr(R, "process_oid",
                        lambda oid, *a, **k: ([{"oid": oid}], None))

    manifest = R.process_unit((0, [1, 2]))

    assert manifest["n_skipped"] == 1
    assert manifest["n_no_allwise"] == 1      # only oid 1, which was actually asked


def _cfg(tmp_path):
    return {"out_dir": str(tmp_path), "minibatch": 500, "features": False,
            "credentials": "creds", "schema": "multisurvey_ztf", "retries": 0,
            "xmatch_url": "http://x", "min_detections": 1}


def test_every_failed_oid_is_written_to_the_errors_sidecar(monkeypatch, tmp_path):
    """The manifest samples 20 errors; the sidecar must name them ALL.

    A unit that hits errors still finishes and still writes its manifest, so the
    resume logic skips it forever: those oids are never retried. If their
    identities only survive in a 20-entry sample, the rest are lost — counted,
    unnamed, and unrecoverable without re-running the whole catalog.
    """
    oids = list(range(1, 26))              # 25 failures, more than the sample cap
    monkeypatch.setattr(R, "_W", {"cfg": _cfg(tmp_path)})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise=set(mb)))

    def _always_fails(oid, *a, **k):
        raise RuntimeError(f"boom {oid}")
    monkeypatch.setattr(R, "process_oid", _always_fails)

    manifest = R.process_unit((0, oids))

    assert manifest["n_errors"] == 25
    assert len(manifest["errors"]) == 20          # the manifest stays a summary

    sidecar = tmp_path / "errors" / "unit_0000000.jsonl"
    lines = sidecar.read_text().strip().splitlines()
    assert len(lines) == 25
    recorded = [json.loads(line) for line in lines]
    assert sorted(r["oid"] for r in recorded) == oids
    assert "boom 7" in next(r["error"] for r in recorded if r["oid"] == 7)


def test_no_errors_file_when_the_unit_is_clean(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_W", {"cfg": _cfg(tmp_path)})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise=set(mb)))
    monkeypatch.setattr(R, "process_oid", lambda oid, *a, **k: ([{"oid": oid}], None))

    R.process_unit((0, [1, 2]))

    assert not (tmp_path / "errors" / "unit_0000000.jsonl").exists()


def test_manifest_separates_the_three_reasons_an_oid_is_skipped(monkeypatch, tmp_path):
    """Only one of the three is worth retrying, so they cannot share a counter.

    no detections   -> nothing to classify, expected, not a problem
    unclassifiable  -> too few real detections after preprocessing, expected
    error           -> the one you want to find and re-run
    """
    monkeypatch.setattr(R, "_W", {"cfg": _cfg(tmp_path)})
    # oid 4 has no detections: absent from the fetched inputs.
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs([1, 2, 3], with_allwise={1, 2, 3}))

    def _outcomes(oid, *a, **k):
        if oid == 2:
            raise RuntimeError("boom")
        if oid == 3:
            return [], None            # unclassifiable: no probability rows
        return [{"oid": oid}], None
    monkeypatch.setattr(R, "process_oid", _outcomes)

    manifest = R.process_unit((0, [1, 2, 3, 4]))

    assert manifest["n_ok"] == 1
    assert manifest["n_errors"] == 1
    assert manifest["n_unclassifiable"] == 1
    assert manifest["n_no_detections"] == 1
    # n_skipped stays the total of the three, so existing readers keep working
    assert manifest["n_skipped"] == 3
