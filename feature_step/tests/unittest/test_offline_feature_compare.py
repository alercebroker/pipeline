"""Tests for features/offline/feature_compare.py — pure, no DB needed."""
import numpy as np
import pandas as pd
import pytest

from features.offline.feature_compare import compare_feature_frames, latest_feature_version


# ---------------------------------------------------------------------------
# Helpers to build minimal DataFrames
# ---------------------------------------------------------------------------

def _ours(rows):
    """rows: list of (name, value, fid_letter_or_None)"""
    return pd.DataFrame(rows, columns=["name", "value", "fid"])


def _theirs(rows):
    """rows: list of (name, value, fid_int)"""
    return pd.DataFrame(rows, columns=["name", "value", "fid"])


# ---------------------------------------------------------------------------
# All-match case
# ---------------------------------------------------------------------------

def test_all_match_exact():
    ours = _ours([("Amplitude", 0.5, "g"), ("Amplitude", 0.6, "r")])
    theirs = _theirs([("Amplitude", 0.5, 1), ("Amplitude", 0.6, 2)])
    merged, summary = compare_feature_frames(ours, theirs)

    assert summary["n_compared"] == 2
    assert summary["match"] == 2
    assert summary["differ"] == 0
    assert summary["only_ours"] == 0
    assert summary["only_theirs"] == 0
    assert all(merged["status"] == "match")


def test_match_within_tolerance():
    ours = _ours([("Amplitude", 0.5000001, "g")])
    theirs = _theirs([("Amplitude", 0.5, 1)])
    _, summary = compare_feature_frames(ours, theirs, rtol=1e-3, atol=1e-6)
    assert summary["match"] == 1
    assert summary["differ"] == 0


# ---------------------------------------------------------------------------
# NaN == NaN counts as match
# ---------------------------------------------------------------------------

def test_nan_nan_match():
    ours = _ours([("IAR_phi", float("nan"), "g")])
    theirs = _theirs([("IAR_phi", float("nan"), 1)])
    merged, summary = compare_feature_frames(ours, theirs)
    assert summary["match"] == 1
    assert summary["differ"] == 0
    assert merged.iloc[0]["status"] == "match"


def test_nan_vs_real_is_differ():
    ours = _ours([("IAR_phi", float("nan"), "g")])
    theirs = _theirs([("IAR_phi", 1.0, 1)])
    _, summary = compare_feature_frames(ours, theirs)
    assert summary["differ"] == 1


# ---------------------------------------------------------------------------
# differ
# ---------------------------------------------------------------------------

def test_differ_beyond_tolerance():
    ours = _ours([("Amplitude", 1.0, "g")])
    theirs = _theirs([("Amplitude", 2.0, 1)])
    merged, summary = compare_feature_frames(ours, theirs)
    assert summary["differ"] == 1
    assert summary["match"] == 0
    row = merged.iloc[0]
    assert row["status"] == "differ"
    assert abs(row["abs_diff"] - 1.0) < 1e-9
    assert abs(row["rel_diff"] - 0.5) < 1e-6  # |1-2|/max(|2|,atol)


# ---------------------------------------------------------------------------
# only_ours / only_theirs
# ---------------------------------------------------------------------------

def test_only_ours():
    ours = _ours([("ExtraFeature", 1.0, "g")])
    theirs = _theirs([])
    merged, summary = compare_feature_frames(ours, theirs)
    assert summary["only_ours"] == 1
    assert summary["only_theirs"] == 0
    assert merged.iloc[0]["status"] == "only_ours"


def test_only_theirs():
    ours = _ours([])
    theirs = _theirs([("ExtraFeature", 1.0, 1)])
    merged, summary = compare_feature_frames(ours, theirs)
    assert summary["only_ours"] == 0
    assert summary["only_theirs"] == 1
    assert merged.iloc[0]["status"] == "only_theirs"


# ---------------------------------------------------------------------------
# Mixed: all four statuses in one frame
# ---------------------------------------------------------------------------

def test_all_four_statuses():
    ours = _ours([
        ("MatchFeat", 1.0, "g"),      # match
        ("NaNFeat", float("nan"), "r"),  # nan-nan match
        ("DiffFeat", 10.0, "g"),     # differ
        ("OnlyOurs", 5.0, "g,r"),    # only_ours
    ])
    theirs = _theirs([
        ("MatchFeat", 1.0, 1),
        ("NaNFeat", float("nan"), 2),
        ("DiffFeat", 20.0, 1),
        ("OnlyTheirs", 99.0, 12),
    ])
    merged, summary = compare_feature_frames(ours, theirs)

    assert summary["match"] == 2
    assert summary["differ"] == 1
    assert summary["only_ours"] == 1
    assert summary["only_theirs"] == 1
    # n_compared is count of pairs present on both sides
    assert summary["n_compared"] == 3  # MatchFeat, NaNFeat, DiffFeat
    assert summary["n_match"] == 2


# ---------------------------------------------------------------------------
# fid letter -> int mapping
# ---------------------------------------------------------------------------

def test_fid_g_maps_to_1():
    ours = _ours([("F", 1.0, "g")])
    theirs = _theirs([("F", 1.0, 1)])
    _, summary = compare_feature_frames(ours, theirs)
    assert summary["match"] == 1


def test_fid_r_maps_to_2():
    ours = _ours([("F", 1.0, "r")])
    theirs = _theirs([("F", 1.0, 2)])
    _, summary = compare_feature_frames(ours, theirs)
    assert summary["match"] == 1


def test_fid_gr_maps_to_12():
    ours = _ours([("F", 1.0, "g,r")])
    theirs = _theirs([("F", 1.0, 12)])
    _, summary = compare_feature_frames(ours, theirs)
    assert summary["match"] == 1


def test_fid_none_maps_to_0():
    ours = _ours([("F", 1.0, None)])
    theirs = _theirs([("F", 1.0, 0)])
    _, summary = compare_feature_frames(ours, theirs)
    assert summary["match"] == 1


# ---------------------------------------------------------------------------
# summary includes only_ours_names and only_theirs_names
# ---------------------------------------------------------------------------

def test_summary_only_names():
    ours = _ours([("OnlyOursA", 1.0, "g"), ("OnlyOursB", 2.0, "r")])
    theirs = _theirs([("OnlyTheirsX", 9.0, 1)])
    _, summary = compare_feature_frames(ours, theirs)

    assert "only_ours_names" in summary
    assert "only_theirs_names" in summary
    assert set(summary["only_ours_names"]) == {"OnlyOursA", "OnlyOursB"}
    assert set(summary["only_theirs_names"]) == {"OnlyTheirsX"}


def test_summary_only_names_sorted():
    ours = _ours([("Z_feat", 1.0, "g"), ("A_feat", 2.0, "r")])
    theirs = _theirs([])
    _, summary = compare_feature_frames(ours, theirs)
    assert summary["only_ours_names"] == sorted(summary["only_ours_names"])


def test_summary_only_names_truncated_to_30():
    # Build 40 unique only_ours features
    rows = [(f"Feat{i:03d}", float(i), "g") for i in range(40)]
    ours = _ours(rows)
    theirs = _theirs([])
    _, summary = compare_feature_frames(ours, theirs)
    assert len(summary["only_ours_names"]) <= 30


# ---------------------------------------------------------------------------
# merged DataFrame columns
# ---------------------------------------------------------------------------

def test_merged_columns():
    ours = _ours([("A", 1.0, "g")])
    theirs = _theirs([("A", 1.0, 1)])
    merged, _ = compare_feature_frames(ours, theirs)
    for col in ["name", "fid_int", "value_ours", "value_theirs", "abs_diff", "rel_diff", "status"]:
        assert col in merged.columns, f"Missing column: {col}"


def test_merged_fid_int_column():
    ours = _ours([("A", 1.0, "g")])
    theirs = _theirs([("A", 1.0, 1)])
    merged, _ = compare_feature_frames(ours, theirs)
    assert merged.iloc[0]["fid_int"] == 1


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------

def test_both_empty():
    ours = _ours([])
    theirs = _theirs([])
    merged, summary = compare_feature_frames(ours, theirs)
    assert summary["n_compared"] == 0
    assert len(merged) == 0


def test_ours_empty():
    ours = _ours([])
    theirs = _theirs([("A", 1.0, 1)])
    _, summary = compare_feature_frames(ours, theirs)
    assert summary["only_theirs"] == 1
    assert summary["n_compared"] == 0


# ---------------------------------------------------------------------------
# latest_feature_version
# ---------------------------------------------------------------------------

_REALISTIC_VERSIONS = [
    "23.12.25",
    "25.0.1a13",
    "26.0.0",
    "27.3.0",
    "27.5.7a32.dev1",
    "lc_classifier_1.2.1-P",
    "lc_classifier_1.2.1-P-transitional",
]


def test_latest_version_realistic_mixed():
    """Realistic mixed list: modern group wins and 27.5.7a32.dev1 > 27.3.0."""
    assert latest_feature_version(_REALISTIC_VERSIONS) == "27.5.7a32.dev1"


def test_latest_version_modern_only():
    versions = ["27.5.2", "27.5.7a32.dev1", "27.3.0"]
    assert latest_feature_version(versions) == "27.5.7a32.dev1"


def test_latest_version_legacy_fallback():
    """Only-legacy list: falls back to legacy, returns a non-None value."""
    versions = ["lc_classifier_1.2.1-P", "lc_classifier_1.2.1-P-transitional"]
    result = latest_feature_version(versions)
    assert result is not None
    assert result in versions


def test_latest_version_empty():
    assert latest_feature_version([]) is None
