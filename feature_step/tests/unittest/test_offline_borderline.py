"""How decided each side was, on the objects where the two disagree.

A rank-1 flip between two classes separated by 0.002 is not the same event as a
flip between 0.9 and 0.05, and counting both as "disagreement" hides which one
happened. This measures the margin on each side and where the other side's
winner sat in this side's own ranking.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPE / "feature_step"))

from features.offline.probability_compare import borderline_report

COLS = ["oid", "classifier_name", "class_name", "probability"]


def _rows(*tuples):
    return pd.DataFrame(list(tuples), columns=COLS)


def test_the_margin_is_the_gap_between_first_and_second():
    ours = _rows((1, "flat", "RSCVn", 0.40, ), (1, "flat", "CV/Nova", 0.38), (1, "flat", "YSO", 0.22))
    legacy = _rows((1, "flat", "CV/Nova", 0.90), (1, "flat", "RSCVn", 0.06), (1, "flat", "YSO", 0.04))

    got = borderline_report(ours, legacy).set_index(["oid", "classifier_name"]).loc[(1, "flat")]

    assert got["margin_ours"] == pytest_approx(0.02)
    assert got["margin_legacy"] == pytest_approx(0.84)


def test_the_other_sides_winner_is_located_in_this_sides_ranking():
    """The question the flip actually raises: was our class legacy's close
    second, or something it had ranked last?"""
    ours = _rows((1, "flat", "RSCVn", 0.40), (1, "flat", "CV/Nova", 0.38), (1, "flat", "YSO", 0.22))
    legacy = _rows((1, "flat", "CV/Nova", 0.90), (1, "flat", "RSCVn", 0.06), (1, "flat", "YSO", 0.04))

    got = borderline_report(ours, legacy).set_index(["oid", "classifier_name"]).loc[(1, "flat")]

    assert got["rank_of_ours_in_legacy"] == 2
    assert got["prob_legacy_for_our_class"] == pytest_approx(0.06)
    assert got["rank_of_legacy_in_ours"] == 2
    assert got["prob_ours_for_legacy_class"] == pytest_approx(0.38)


def test_a_class_the_other_side_never_scored_is_missing_not_zero():
    """Reporting 0.0 for a class legacy never emitted would read as 'legacy ruled
    it out', which is a different claim from 'legacy never considered it'."""
    ours = _rows((1, "flat", "TDE", 0.7), (1, "flat", "SNIa", 0.3))
    legacy = _rows((1, "flat", "SNIa", 0.6), (1, "flat", "SNII", 0.4))

    got = borderline_report(ours, legacy).set_index(["oid", "classifier_name"]).loc[(1, "flat")]

    assert np.isnan(got["prob_legacy_for_our_class"])
    assert np.isnan(got["rank_of_ours_in_legacy"])


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=1e-9)
