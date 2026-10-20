"""compare_rows: shard vs stored probability rows, the pure half of the validation script."""
import sys
from pathlib import Path

import pandas as pd

PIPE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPE / "feature_step" / "scripts"))

import offline_reclassify_compare as C  # noqa: E402

COLS = ["oid", "classifier_id", "class_id", "probability", "ranking", "lastmjd"]


def _rows(records):
    """Rows in the shard / stored layout."""
    return pd.DataFrame(records, columns=COLS)


def test_identical_rows_are_ok_within_float32_rounding():
    """A stored float32 value differs from the computed float64 one by well under 1e-6."""
    shard = _rows([(1, 5, 0, 0.292992, 1, 60896.5), (1, 5, 1, 0.707008, 2, 60896.5)])
    stored = _rows([(1, 5, 0, 0.29299200773, 1, 60896.5), (1, 5, 1, 0.70700800419, 2, 60896.5)])
    out = C.compare_rows(shard, stored, tol=1e-6)
    assert out["verdict"] == "ok"
    assert out["oids_compared"] == 1 and out["rows_shard"] == 2 and out["rows_db"] == 2
    assert out["max_abs_dprob"] < 1e-6


def test_a_probability_gap_a_rank_change_or_a_missing_row_is_different():
    """Each check fails the verdict on its own and is counted on its own line."""
    shard = _rows([(1, 5, 0, 0.5, 1, 60896.5), (1, 5, 1, 0.5, 1, 60896.5), (2, 5, 0, 0.9, 1, 60000.0)])
    stored = _rows([(1, 5, 0, 0.6, 1, 60896.5), (1, 5, 1, 0.5, 2, 60896.5)])   # oid 2 absent
    out = C.compare_rows(shard, stored, tol=1e-6)
    assert out["verdict"] == "DIFFERENT"
    assert out["rows_over_tol"] == 1          # 0.5 vs 0.6
    assert out["rank_mismatches"] == 1        # 1 vs 2 on class 1
    assert out["missing_in_db"] == 1          # oid 2
    assert abs(out["max_abs_dprob"] - 0.1) < 1e-12


def test_extra_stored_rows_are_reported():
    """A stored row the shard did not produce means a class or classifier was lost."""
    shard = _rows([(1, 5, 0, 0.5, 1, 60896.5)])
    stored = _rows([(1, 5, 0, 0.5, 1, 60896.5), (1, 5, 1, 0.5, 1, 60896.5)])
    out = C.compare_rows(shard, stored, tol=1e-6)
    assert out["verdict"] == "DIFFERENT" and out["extra_in_db"] == 1
