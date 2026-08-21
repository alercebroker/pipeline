"""Aggregate rank-1 agreement between two *stored* probability tables.

The per-oid comparator in probability_compare.py rebuilds the offline side from a
BHRF OutputDTO, so a study costs one full pipeline run per object -- which is why
the previous one covered 100 oids. Once a batch run has loaded its probabilities
into <schema>.probability, both sides are just DB rows and the same study is a
join over tens of thousands of objects.
"""
import sys
from pathlib import Path

import pandas as pd

PIPE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPE / "feature_step"))

from features.offline.probability_compare import rank1_agreement

COLS = ["oid", "classifier_name", "class_name", "probability", "ranking"]


def _rows(*tuples):
    return pd.DataFrame(list(tuples), columns=COLS)


def test_agreement_is_counted_per_classifier():
    ours = _rows(
        (1, "flat", "QSO", 0.7, 1), (1, "flat", "AGN", 0.3, 2),
        (1, "top", "Stochastic", 0.9, 1), (1, "top", "Periodic", 0.1, 2),
        (2, "flat", "LPV", 0.6, 1), (2, "flat", "QSO", 0.4, 2),
    )
    legacy = _rows(
        (1, "flat", "QSO", 0.65, 1), (1, "flat", "AGN", 0.35, 2),
        (1, "top", "Periodic", 0.8, 1), (1, "top", "Stochastic", 0.2, 2),
        (2, "flat", "CV/Nova", 0.5, 1), (2, "flat", "LPV", 0.5, 2),
    )
    _, summary = rank1_agreement(ours, legacy)

    assert summary["by_classifier"]["flat"] == {"n_both": 2, "n_agree": 1, "rate": 0.5}
    assert summary["by_classifier"]["top"] == {"n_both": 1, "n_agree": 0, "rate": 0.0}


def test_an_oid_on_only_one_side_is_reported_but_not_scored():
    """Legacy has no row for objects it never classified. Counting those as
    disagreements would report a failure of our pipeline where there is simply
    nothing to compare against."""
    ours = _rows(
        (1, "flat", "QSO", 0.7, 1),
        (2, "flat", "LPV", 0.6, 1),
    )
    legacy = _rows((1, "flat", "QSO", 0.7, 1))

    per_oid, summary = rank1_agreement(ours, legacy)

    assert summary["by_classifier"]["flat"] == {"n_both": 1, "n_agree": 1, "rate": 1.0}
    assert summary["n_only_ours"] == 1
    assert summary["n_only_legacy"] == 0
    assert set(per_oid["oid"]) == {1}


def test_the_stored_ranking_column_is_not_trusted():
    """Two writers produced these tables years apart. Deriving rank 1 from the
    probabilities themselves means the comparison cannot be corrupted by a
    ranking convention that differs between them."""
    ours = _rows((1, "flat", "QSO", 0.9, 2), (1, "flat", "AGN", 0.1, 1))
    legacy = _rows((1, "flat", "QSO", 0.8, 2), (1, "flat", "AGN", 0.2, 1))

    per_oid, summary = rank1_agreement(ours, legacy)

    assert per_oid.loc[0, "class_ours"] == "QSO"
    assert per_oid.loc[0, "class_legacy"] == "QSO"
    assert summary["by_classifier"]["flat"]["n_agree"] == 1


def test_a_probability_tie_is_broken_the_same_way_on_both_sides():
    """A tie picked by row order would score an agreement or a disagreement
    depending on how the DB happened to return the rows."""
    ours = _rows((1, "flat", "QSO", 0.5, 1), (1, "flat", "AGN", 0.5, 1))
    legacy = _rows((1, "flat", "AGN", 0.5, 1), (1, "flat", "QSO", 0.5, 1))

    per_oid, summary = rank1_agreement(ours, legacy)

    assert per_oid.loc[0, "class_ours"] == "AGN"      # class_name ascending
    assert per_oid.loc[0, "class_legacy"] == "AGN"
    assert summary["by_classifier"]["flat"]["n_agree"] == 1
