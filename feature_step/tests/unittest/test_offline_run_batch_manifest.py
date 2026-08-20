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


def _inputs(oids, with_allwise, matches=None):
    """({oid: (message, refs, allwise)}, matches) — allwise empty where absent.

    fetch_minibatch returns the raw Xwave matches beside the inputs so the unit
    can persist the crossmatch link rows it already paid for.
    """
    cols = ["oid", "W1", "W2", "W3", "W4"]
    out = {}
    for oid in oids:
        allwise = (pd.DataFrame([{"oid": oid, "W1": 1.0, "W2": 2.0, "W3": 3.0, "W4": 4.0}])
                   if oid in with_allwise else pd.DataFrame(columns=cols))
        out[oid] = ({"oid": oid}, pd.DataFrame(), allwise)
    if matches is None:
        matches = [{"oid": str(o)} for o in oids if o in with_allwise]
    return out, matches


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


def _cfg_db(tmp_path, **over):
    cfg = _cfg(tmp_path)
    cfg.update({"load_db": True, "write_credentials": "write-creds"})
    cfg.update(over)
    return cfg


def test_load_db_writes_both_tables_and_records_the_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_W", {"cfg": _cfg_db(tmp_path, features=True)})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise=set(mb)))
    monkeypatch.setattr(R, "process_oid", lambda oid, *a, **k: (
        [{"oid": oid}], pd.DataFrame({"oid": [oid], "sid": [0], "feature_id": [1],
                                      "band": [1], "version": [1], "value": [0.5]})))

    seen = {}
    monkeypatch.setattr(R, "write_probabilities", lambda rows, creds, **kw: (
        seen.update(prob=(len(rows), creds, kw.get("execute"))), {"written": len(rows)})[1])
    monkeypatch.setattr(R, "write_features", lambda frame, creds, **kw: (
        seen.update(feat=(len(frame), creds, kw.get("execute"))), {"written": len(frame)})[1])
    monkeypatch.setattr(R, "persist_matches",
                        lambda matches, creds, **kw: {"written": len(matches)})

    manifest = R.process_unit((0, [1, 2, 3]))

    assert seen["prob"] == (3, "write-creds", True)   # ONE call for the whole unit
    assert seen["feat"] == (3, "write-creds", True)
    assert manifest["db_prob_rows"] == 3
    assert manifest["db_feat_rows"] == 3


def test_no_db_write_when_load_db_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_W", {"cfg": _cfg(tmp_path)})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise=set(mb)))
    monkeypatch.setattr(R, "process_oid", lambda oid, *a, **k: ([{"oid": oid}], None))

    def _boom(*a, **k):
        raise AssertionError("must not touch the DB without --load-db")
    monkeypatch.setattr(R, "write_probabilities", _boom)
    monkeypatch.setattr(R, "write_features", _boom)

    manifest = R.process_unit((0, [1, 2]))
    assert manifest["db_prob_rows"] == 0


def test_a_failed_db_load_leaves_the_unit_unmarked(monkeypatch, tmp_path):
    """The manifest is the done-marker, so it must land AFTER the commit.

    If it were written first, a unit whose rows never reached the database would
    be treated as finished forever and the resume logic would skip it. Failing
    with no manifest is what makes the rerun redo it — safe, because the upsert
    is idempotent.
    """
    monkeypatch.setattr(R, "_W", {"cfg": _cfg_db(tmp_path)})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise=set(mb)))
    monkeypatch.setattr(R, "process_oid", lambda oid, *a, **k: ([{"oid": oid}], None))

    def _db_down(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(R, "write_probabilities", _db_down)

    with pytest.raises(RuntimeError, match="connection refused"):
        R.process_unit((0, [1, 2]))

    assert not (tmp_path / "manifests" / "unit_0000000.json").exists()


def test_load_db_persists_the_units_crossmatch_in_one_call(monkeypatch, tmp_path):
    """The crossmatch is the feature step's own output, and we already paid for it.

    In the live pipeline the step produces the xmatch to the scribe, which upserts
    <schema>.xmatch. Offline we replace that step, so dropping the matches would
    leave 7.45M objects classified with no record of which AllWISE source they
    were matched against, or how far away it was. One call per unit, like the
    other two tables -- not one per minibatch and not one per oid.
    """
    cfg = _cfg_db(tmp_path)
    cfg["minibatch"] = 2                       # 4 oids -> 2 minibatches
    monkeypatch.setattr(R, "_W", {"cfg": cfg})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise=set(mb)))
    monkeypatch.setattr(R, "process_oid", lambda oid, *a, **k: ([{"oid": oid}], None))
    monkeypatch.setattr(R, "write_probabilities",
                        lambda rows, creds, **kw: {"written": len(rows)})

    seen = []
    monkeypatch.setattr(R, "persist_matches", lambda matches, creds, **kw: (
        seen.append((len(matches), creds, kw.get("execute"))), {"written": len(matches)})[1])

    manifest = R.process_unit((0, [1, 2, 3, 4]))

    assert seen == [(4, "write-creds", True)]   # ONE call, both minibatches in it
    assert manifest["db_xmatch_rows"] == 4


def test_no_crossmatch_write_when_load_db_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_W", {"cfg": _cfg(tmp_path)})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise=set(mb)))
    monkeypatch.setattr(R, "process_oid", lambda oid, *a, **k: ([{"oid": oid}], None))

    def _boom(*a, **k):
        raise AssertionError("must not touch the DB without --load-db")
    monkeypatch.setattr(R, "persist_matches", _boom)

    assert R.process_unit((0, [1, 2]))["db_xmatch_rows"] == 0


def test_no_crossmatch_write_when_the_run_reads_allwise_from_the_db(monkeypatch, tmp_path):
    """Without --xmatch-url there are no matches to persist.

    persist_matches([]) would still open a connection to write nothing, once per
    unit for the whole run.
    """
    cfg = _cfg_db(tmp_path)
    cfg["xmatch_url"] = None
    monkeypatch.setattr(R, "_W", {"cfg": cfg})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise=set(mb), matches=[]))
    monkeypatch.setattr(R, "process_oid", lambda oid, *a, **k: ([{"oid": oid}], None))
    monkeypatch.setattr(R, "write_probabilities",
                        lambda rows, creds, **kw: {"written": len(rows)})

    def _boom(*a, **k):
        raise AssertionError("nothing to persist; must not connect")
    monkeypatch.setattr(R, "persist_matches", _boom)

    assert R.process_unit((0, [1, 2]))["db_xmatch_rows"] == 0


def test_failed_units_make_the_run_exit_non_zero():
    """A run that lost units is not a successful run.

    Returning 0 regardless is what lets a supervisor, a cron entry or a CI step
    read a run that dropped half its units as clean. The units are recoverable
    -- they were left unmarked, so a rerun redoes them -- but only if somebody
    is told to rerun.
    """
    assert R.exit_code(n_failed_units=3, n_oid_errors=0) != 0


def test_a_clean_run_exits_zero():
    assert R.exit_code(n_failed_units=0, n_oid_errors=0) == 0


def test_per_oid_errors_alone_do_not_fail_the_run():
    """Individual bad objects are an expected outcome at this scale.

    They are counted, named in errors/unit_*.jsonl and retried explicitly; a
    handful out of 26M must not mark the whole run as failed, or the exit code
    stops meaning anything.
    """
    assert R.exit_code(n_failed_units=0, n_oid_errors=250) == 0


def test_no_shards_skips_the_parquet_but_still_marks_the_unit_done(monkeypatch, tmp_path):
    """With --load-db the database holds the answer and nothing reads the shards.

    ~68 GB of parquet across the run that no tool opens: resume keys off the
    manifest, not the shards, and there is no loader that could read them back.
    The manifest and the error sidecar must still be written -- they are the
    resume machinery and the only record of which oids failed.
    """
    cfg = _cfg_db(tmp_path, features=True)
    cfg["no_shards"] = True
    monkeypatch.setattr(R, "_W", {"cfg": cfg})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise=set(mb)))
    monkeypatch.setattr(R, "process_oid", lambda oid, *a, **k: (
        [{"oid": oid}], pd.DataFrame({"oid": [oid], "sid": [0], "feature_id": [1],
                                      "band": [1], "version": [1], "value": [0.5]})))
    monkeypatch.setattr(R, "write_probabilities",
                        lambda rows, creds, **kw: {"written": len(rows)})
    monkeypatch.setattr(R, "write_features",
                        lambda frame, creds, **kw: {"written": len(frame)})
    monkeypatch.setattr(R, "persist_matches",
                        lambda matches, creds, **kw: {"written": len(matches)})

    manifest = R.process_unit((0, [1, 2, 3]))

    assert not list(tmp_path.glob("probabilities/*.parquet"))
    assert not list(tmp_path.glob("features/*.parquet"))
    assert (tmp_path / "manifests" / "unit_0000000.json").exists()
    assert manifest["db_prob_rows"] == 3      # the rows still went to the DB


def test_shards_are_written_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_W", {"cfg": _cfg(tmp_path)})
    monkeypatch.setattr(R, "fetch_minibatch",
                        lambda mb, cfg: _inputs(mb, with_allwise=set(mb)))
    monkeypatch.setattr(R, "process_oid", lambda oid, *a, **k: ([{"oid": oid}], None))

    R.process_unit((0, [1, 2]))
    assert list(tmp_path.glob("probabilities/*.parquet"))


def test_no_shards_is_refused_without_load_db():
    """Then the run would compute 26M objects and keep none of the results."""
    assert R.shards_would_be_lost(no_shards=True, load_db=False) is True
    assert R.shards_would_be_lost(no_shards=True, load_db=True) is False
    assert R.shards_would_be_lost(no_shards=False, load_db=False) is False
