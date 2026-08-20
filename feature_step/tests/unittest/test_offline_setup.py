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
