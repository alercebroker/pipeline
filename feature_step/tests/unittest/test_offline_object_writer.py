"""Unit tests for object_writer — the ztf_object colour backfill. No real DB.

The live feature step sends an `update-ztf-object-features` command per object
that the scribe turns into UPDATE ztf_object SET g_r_max, g_r_mean, g_r_max_corr,
g_r_mean_corr. The offline run never did that; this module rebuilds those four
columns from the `feature` rows already written. The connection is faked so the
SQL and the parameters sent per range are what is asserted.
"""
import json

import numpy as np
import pytest

from features.offline import object_writer


# --------------------------------------------------------------------------- #
#  feature ids come from the DB LUT, by name
# --------------------------------------------------------------------------- #
def test_resolve_color_feature_ids_maps_each_column_to_its_lut_id():
    lut = {7: "g-r_mean", 3: "g-r_max", 11: "g-r_mean_corr", 5: "g-r_max_corr",
           9: "Amplitude"}
    ids = object_writer.resolve_color_feature_ids(lut)
    assert ids == {"g_r_max": 3, "g_r_mean": 7, "g_r_max_corr": 5, "g_r_mean_corr": 11}


def test_resolve_color_feature_ids_names_the_missing_feature():
    lut = {0: "g-r_mean", 1: "g-r_max", 2: "g-r_mean_corr"}
    with pytest.raises(LookupError, match="g-r_max_corr"):
        object_writer.resolve_color_feature_ids(lut)


# --------------------------------------------------------------------------- #
#  ranges are slices of the sorted, deduplicated oid array
# --------------------------------------------------------------------------- #
def test_make_ranges_slices_sorted_unique_oids_by_chunk_size():
    oids = np.array([50, 10, 30, 20, 40, 30], dtype=np.int64)
    assert object_writer.make_ranges(oids, chunk_size=2) == [(10, 20), (30, 40), (50, 50)]


def test_make_ranges_of_nothing_is_empty():
    assert object_writer.make_ranges(np.empty(0, dtype=np.int64), chunk_size=5) == []


def test_make_ranges_rejects_a_non_positive_chunk_size():
    with pytest.raises(ValueError):
        object_writer.make_ranges(np.array([1, 2], dtype=np.int64), chunk_size=0)


# --------------------------------------------------------------------------- #
#  one UPDATE ... FROM per range
# --------------------------------------------------------------------------- #
def test_build_backfill_sql_updates_the_four_colours_from_band_12_rows():
    ids = {"g_r_max": 1, "g_r_mean": 0, "g_r_max_corr": 3, "g_r_mean_corr": 2}
    sql = object_writer.build_backfill_sql("multisurvey_ztf", ids)

    assert "UPDATE multisurvey_ztf.ztf_object" in sql
    assert "FROM multisurvey_ztf.feature" in sql
    assert f"band = {object_writer.PAIR_BAND}" in sql
    for column, feature_id in ids.items():
        assert f"FILTER (WHERE feature_id = {feature_id}) AS {column}" in sql
    assert "feature_id IN (0, 1, 2, 3)" in sql
    for param in (":lo", ":hi", ":sid"):
        assert param in sql


def test_build_backfill_sql_keeps_only_the_latest_feature_run_per_oid():
    """A colour computed in an older run but not in the latest one must become
    NULL, as the step would have sent None -- see feature-rows-superseded."""
    ids = {"g_r_max": 1, "g_r_mean": 0, "g_r_max_corr": 3, "g_r_mean_corr": 2}
    sql = object_writer.build_backfill_sql("s", ids)
    assert "max(updated_date) OVER (PARTITION BY oid)" in sql


# --------------------------------------------------------------------------- #
#  the runner: sequential, resumable, dry by default
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class _FakeConn:
    def __init__(self):
        self.executed = []      # (sql, params)
        self.commits = 0

    def execute(self, sql, params=None):
        self.executed.append((str(sql), params))
        return _FakeResult(rowcount=params["hi"] - params["lo"] + 1)

    def commit(self):
        self.commits += 1


def test_run_backfill_dry_run_touches_nothing_and_reports_the_plan(tmp_path):
    progress = tmp_path / "progress.jsonl"
    conn = _FakeConn()
    ranges = [(10, 20), (30, 40)]

    summary = object_writer.run_backfill(
        conn, "UPDATE ...", ranges, sid=0, progress_path=progress, execute=False)

    assert conn.executed == []
    assert summary == {"executed": False, "ranges": 2, "skipped": 0, "updated": 0}
    assert not progress.exists()


def test_run_backfill_executes_one_statement_per_range_and_commits_each(tmp_path):
    progress = tmp_path / "progress.jsonl"
    conn = _FakeConn()
    ranges = [(10, 20), (30, 40)]

    summary = object_writer.run_backfill(
        conn, "UPDATE :lo :hi :sid", ranges, sid=0, progress_path=progress, execute=True)

    assert [p for _, p in conn.executed] == [{"lo": 10, "hi": 20, "sid": 0},
                                             {"lo": 30, "hi": 40, "sid": 0}]
    assert conn.commits == 2
    assert summary == {"executed": True, "ranges": 2, "skipped": 0, "updated": 22}

    lines = [json.loads(l) for l in progress.read_text().splitlines()]
    assert [(l["chunk"], l["lo"], l["hi"], l["updated"]) for l in lines] == [
        (0, 10, 20, 11), (1, 30, 40, 11)]


def test_run_backfill_resumes_past_ranges_already_in_the_progress_file(tmp_path):
    progress = tmp_path / "progress.jsonl"
    progress.write_text(json.dumps({"chunk": 0, "lo": 10, "hi": 20, "updated": 11,
                                    "seconds": 0.1}) + "\n")
    conn = _FakeConn()

    summary = object_writer.run_backfill(
        conn, "UPDATE", [(10, 20), (30, 40)], sid=0, progress_path=progress, execute=True)

    assert [p for _, p in conn.executed] == [{"lo": 30, "hi": 40, "sid": 0}]
    assert summary == {"executed": True, "ranges": 2, "skipped": 1, "updated": 11}


def test_run_backfill_refuses_a_progress_file_from_a_different_range_plan(tmp_path):
    progress = tmp_path / "progress.jsonl"
    progress.write_text(json.dumps({"chunk": 0, "lo": 1, "hi": 2, "updated": 0,
                                    "seconds": 0.1}) + "\n")
    conn = _FakeConn()

    with pytest.raises(ValueError, match="chunk 0"):
        object_writer.run_backfill(
            conn, "UPDATE", [(10, 20)], sid=0, progress_path=progress, execute=True)
    assert conn.executed == []


def test_run_backfill_max_chunks_stops_after_that_many_new_ranges(tmp_path):
    progress = tmp_path / "progress.jsonl"
    conn = _FakeConn()

    summary = object_writer.run_backfill(
        conn, "UPDATE", [(10, 20), (30, 40), (50, 60)], sid=0,
        progress_path=progress, execute=True, max_chunks=2)

    assert len(conn.executed) == 2
    assert summary["ranges"] == 3
    # a rerun without the cap picks up the third
    conn2 = _FakeConn()
    summary2 = object_writer.run_backfill(
        conn2, "UPDATE", [(10, 20), (30, 40), (50, 60)], sid=0,
        progress_path=progress, execute=True)
    assert [p["lo"] for _, p in conn2.executed] == [50]
    assert summary2["skipped"] == 2


# --------------------------------------------------------------------------- #
#  the CLI wires: oid file -> ranges, DB LUT -> ids, and dry-runs by default
# --------------------------------------------------------------------------- #
def test_cli_dry_run_resolves_ids_from_the_db_lut_and_executes_nothing(tmp_path, monkeypatch, capsys):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import offline_backfill_object_colors as cli

    oid_file = tmp_path / "run.npy"
    np.save(oid_file, np.array([30, 10, 20, 40, 50], dtype=np.int64))

    conn = _FakeConn()
    monkeypatch.setattr(cli, "connect", lambda credentials, timeout_s: conn)
    monkeypatch.setattr(cli.db, "fetch_feature_name_lut",
                        lambda credentials, sid, schema: {0: "g-r_mean", 1: "g-r_max",
                                                          2: "g-r_mean_corr", 3: "g-r_max_corr"})

    rc = cli.main(["--oid-file", str(oid_file), "--credentials", "creds.json",
                   "--out-dir", str(tmp_path / "out"), "--chunk-size", "2"])

    assert rc == 0
    assert conn.executed == []
    out = capsys.readouterr().out
    assert "3 range(s)" in out
    assert "g_r_max_corr=3" in out
    assert "dry run" in out
    assert not (tmp_path / "out" / "progress.jsonl").exists()
