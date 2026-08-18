"""Tests for features/offline/probability_compare.py — pure, no DB/model needed."""
import pandas as pd
import pytest

from features.offline.probability_compare import (
    BHRF_CLASSIFIER_NAMES,
    compare_probability_frames,
)

FLAT = "lc_classifier_BHRF_forced_phot"
TOP = "lc_classifier_BHRF_forced_phot_top"


def _offline(mapping):
    """mapping: {classifier_name: {class_name: prob}} -> {name: Series}."""
    return {c: pd.Series(d) for c, d in mapping.items()}


def _stored(rows):
    """rows: list of (classifier_name, class_name, probability, ranking)."""
    return pd.DataFrame(rows, columns=["classifier_name", "class_name", "probability", "ranking"])


def test_head_names_match_taxonomy_order():
    # flat, top, transient, stochastic, periodic (ids 5..9)
    assert BHRF_CLASSIFIER_NAMES[0] == FLAT
    assert BHRF_CLASSIFIER_NAMES[1] == TOP
    assert len(BHRF_CLASSIFIER_NAMES) == 5


def test_all_match_and_passed():
    offline = _offline({FLAT: {"QSO": 0.7, "SNIa": 0.3}})
    stored = _stored([(FLAT, "QSO", 0.7, 1), (FLAT, "SNIa", 0.3, 2)])
    merged, summary = compare_probability_frames(offline, stored)
    assert summary["match"] == 2
    assert summary["differ"] == 0
    assert summary["only_offline"] == 0 and summary["only_stored"] == 0
    assert summary["rank1_agree"] == summary["rank1_total"] == 1
    assert summary["passed"] is True


def test_within_tolerance_matches():
    offline = _offline({FLAT: {"QSO": 0.7003, "SNIa": 0.2997}})
    stored = _stored([(FLAT, "QSO", 0.7, 1), (FLAT, "SNIa", 0.3, 2)])
    _, summary = compare_probability_frames(offline, stored, rtol=1e-2, atol=1e-3)
    assert summary["differ"] == 0
    assert summary["passed"] is True


def test_differ_beyond_tolerance_fails():
    offline = _offline({FLAT: {"QSO": 0.9, "SNIa": 0.1}})
    stored = _stored([(FLAT, "QSO", 0.5, 1), (FLAT, "SNIa", 0.5, 2)])
    merged, summary = compare_probability_frames(offline, stored)
    assert summary["differ"] == 2
    assert summary["passed"] is False
    row = merged[merged["class_name"] == "QSO"].iloc[0]
    assert abs(row["abs_diff"] - 0.4) < 1e-9


def test_rank1_disagreement_fails_even_if_probs_close_only_on_some():
    # offline top = SNIa, stored rank1 = QSO -> rank-1 disagree
    offline = _offline({FLAT: {"QSO": 0.4, "SNIa": 0.6}})
    stored = _stored([(FLAT, "QSO", 0.6, 1), (FLAT, "SNIa", 0.4, 2)])
    _, summary = compare_probability_frames(offline, stored)
    assert summary["rank1"][FLAT]["offline"] == "SNIa"
    assert summary["rank1"][FLAT]["stored"] == "QSO"
    assert summary["rank1"][FLAT]["agree"] is False
    assert summary["passed"] is False


def test_only_offline_and_only_stored():
    offline = _offline({FLAT: {"QSO": 0.5, "NewClass": 0.5}})
    stored = _stored([(FLAT, "QSO", 0.5, 1), (FLAT, "GoneClass", 0.5, 2)])
    merged, summary = compare_probability_frames(offline, stored)
    assert summary["only_offline"] == 1  # NewClass
    assert summary["only_stored"] == 1   # GoneClass
    assert summary["passed"] is False
    statuses = dict(zip(merged["class_name"], merged["status"]))
    assert statuses["NewClass"] == "only_offline"
    assert statuses["GoneClass"] == "only_stored"


def test_multiple_heads():
    offline = _offline({
        FLAT: {"QSO": 0.8, "SNIa": 0.2},
        TOP: {"Stochastic": 0.9, "Transient": 0.1},
    })
    stored = _stored([
        (FLAT, "QSO", 0.8, 1), (FLAT, "SNIa", 0.2, 2),
        (TOP, "Stochastic", 0.9, 1), (TOP, "Transient", 0.1, 2),
    ])
    _, summary = compare_probability_frames(offline, stored)
    assert summary["n_compared"] == 4
    assert summary["rank1_total"] == 2
    assert summary["rank1_agree"] == 2
    assert summary["passed"] is True


def test_merged_columns():
    offline = _offline({FLAT: {"QSO": 1.0}})
    stored = _stored([(FLAT, "QSO", 1.0, 1)])
    merged, _ = compare_probability_frames(offline, stored)
    for col in ("classifier_name", "class_name", "prob_offline", "prob_stored",
                "abs_diff", "rel_diff", "status"):
        assert col in merged.columns
