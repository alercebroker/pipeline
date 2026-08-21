"""The setup script's decision logic: what is already done, and what blocks a run.

Everything else in it is I/O against the DB, S3 and Xwave; these are the pure
parts, and they are the ones that decide whether the operator is told the
machine is ready.
"""
import sys
from pathlib import Path

import pytest

PIPE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPE / "feature_step" / "scripts"))

import offline_setup as S


def test_md5_mismatch_is_reported_not_ignored(tmp_path):
    """A truncated download is the failure mode this exists for.

    MODEL_PATH pointing at half a pickle fails much later, inside the model
    loader, with an error that says nothing about the download.
    """
    f = tmp_path / "model.pkl"
    f.write_bytes(b"not the model")
    assert S.verify_md5(f, "95e8e9f18fde62f22025e31a88ad81fa") is False


def test_md5_match_is_accepted(tmp_path):
    import hashlib
    f = tmp_path / "model.pkl"
    f.write_bytes(b"some bytes")
    assert S.verify_md5(f, hashlib.md5(b"some bytes").hexdigest()) is True


def test_a_missing_step_blocks_the_run():
    results = [S.Result("db", S.OK, ""), S.Result("creds", S.MISSING, "no password")]
    assert S.is_ready(results) is False


def test_work_done_now_still_counts_as_ready():
    """DONE means the script fixed it on this pass; that is not a blocker."""
    results = [S.Result("model", S.DONE, "downloaded"), S.Result("db", S.OK, "")]
    assert S.is_ready(results) is True


def test_a_failed_step_blocks_the_run():
    results = [S.Result("xwave", S.FAIL, "connection refused")]
    assert S.is_ready(results) is False


def test_missing_privileges_names_what_is_absent():
    got = S.missing_privileges({"USAGE": True, "detection": False,
                                "feature INSERT": True, "object": False})
    assert got == ["detection", "object"]


def test_nothing_missing_is_an_empty_list():
    assert S.missing_privileges({"USAGE": True, "detection": True}) == []


def test_the_read_check_covers_the_tables_the_run_actually_reads():
    """The grants were issued for the three write tables and the run still could
    not start: every input table -- detections, forced photometry, the object
    list, the LUTs -- needs SELECT too, and nothing checked for it.
    """
    for t in ("detection", "forced_photometry", "object", "feature_name_lut",
              "taxonomy", "ztf_reference"):
        assert t in S.READ_TABLES


def test_a_step_that_raises_becomes_a_failed_row_not_a_traceback():
    """The script exists to report what is wrong; a traceback reports one thing
    and abandons every check after it.

    Real case: MODEL_PATH defaulted under /data, which was not writable, so the
    model step raised PermissionError and the oid list, Xwave and the seeds were
    never reported at all.
    """
    def _boom():
        raise PermissionError("[Errno 13] Permission denied: '/data'")

    got = S.safe_step("modelo BHRF 2.1.0", _boom)
    assert got.status == S.FAIL
    assert got.name == "modelo BHRF 2.1.0"
    assert "Permission denied" in got.detail


def test_a_step_that_succeeds_passes_its_result_through():
    ok = S.Result("modelo", S.OK, "ya estaba")
    assert S.safe_step("modelo", lambda: ok) is ok


def test_default_paths_never_require_root():
    """Both defaults live beside the code, which is writable by whoever cloned
    it. /data was invented and is root-owned on a normal host."""
    assert "/data" not in str(S.DEFAULT_MODEL_PATH)
    assert "/data" not in str(S.DEFAULT_OID_FILE)
    assert str(S.DEFAULT_MODEL_PATH).startswith(str(S.OFFLINE))
    assert str(S.DEFAULT_OID_FILE).startswith(str(S.OFFLINE))


def test_the_oid_list_lands_where_the_run_will_look_for_it(tmp_path, monkeypatch):
    """np.save appends .npy unless the name already ends in it.

    The write is staged through a temporary file so an interrupted scan cannot
    leave a half-written list behind that the next pass would trust. Name that
    file `run.npy.tmp` and numpy silently writes `run.npy.tmp.npy` instead, so
    the rename finds nothing and an hour of scanning `object` is thrown away.
    """
    import types
    import numpy as np

    stub = types.ModuleType("offline_run_batch")
    stub.select_oids = lambda credentials, min_n_det: np.array([7, 11, 13], dtype=np.int64)
    monkeypatch.setitem(sys.modules, "offline_run_batch", stub)

    out = tmp_path / "oids" / "run.npy"
    res = S.step_oids(tmp_path / "credentials.json", out, 2, check_only=False)

    assert res.status == S.DONE
    assert np.load(out).tolist() == [7, 11, 13]
    assert sorted(p.name for p in out.parent.iterdir()) == ["run.npy"]
