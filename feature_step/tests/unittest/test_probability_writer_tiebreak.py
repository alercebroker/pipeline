"""Unique ranking on tied probabilities in probability_writer.build_probability_rows.

One test per criterion of docs/superpowers/specs/2026-10-22-offline-reclassify-ranking-tiebreak-design.md.
Mirrors the live-step fix (lc_classification_multisurvey_step, rank(method="first")).
"""
import pandas as pd  # frames the writer consumes

from alerce_classifiers.base.dto import OutputDTO  # the writer's input type
from features.offline import probability_writer as pw  # unit under test
from features.offline.classifier_taxonomy_lut import TAXONOMY_LUT  # class names per classifier id

OID = 36028941624528297  # any oid; the writer only copies it into the rows
LASTMJD = 60000.0  # any value; the writer only copies it into the rows

# {classifier_id: {class_name: class_id}}, what db.fetch_taxonomy_maps returns at runtime.
TAX_MAPS = {cid: {name: idx for idx, name in enumerate(names)} for cid, names in TAXONOMY_LUT.items()}

TRANSIENT = TAXONOMY_LUT[7]  # 6 classes (SESN, SLSN, SNII, SNIIn, SNIa, TDE): the head with ties in these tests
TIED_PROBS = [0.3, 0.3, 0.2, 0.1, 0.1, 0.05]  # two ties: columns 0 == 1 at the top, columns 3 == 4 in the middle


def _frame(classifier_id, probs):
    """One-oid frame whose columns are the head's class names, in taxonomy (model) order."""
    return pd.DataFrame([dict(zip(TAXONOMY_LUT[classifier_id], probs))], index=[OID])  # column order = model order


def _decreasing(classifier_id):
    """One-oid frame with strictly decreasing probabilities, 1, 1/2, 1/3, ...: no ties."""
    n = len(TAXONOMY_LUT[classifier_id])  # classes of the head
    return _frame(classifier_id, [1.0 / (i + 1) for i in range(n)])  # strictly decreasing left to right


def _rows_of_head(frame_7):
    """Rows of classifier 7 for a DTO that carries `frame_7` as its Transient child."""
    dto = OutputDTO(_decreasing(5), {"top": _decreasing(6), "children": {"Transient": frame_7}})  # heads 5, 6, 7
    rows = pw.build_probability_rows(dto, OID, LASTMJD, TAX_MAPS)  # all rows of the three heads
    return [r for r in rows if r["classifier_id"] == 7]  # only the head under test


def test_tied_probabilities_get_a_permutation_of_1_to_n():
    """Criterion 1: with ties, the rankings of a head are exactly {1..N}, no repeat."""
    rows = _rows_of_head(_frame(7, TIED_PROBS))  # 6 rows, two tie pairs
    assert sorted(r["ranking"] for r in rows) == [1, 2, 3, 4, 5, 6]  # a permutation, not [1, 1, 2, 3, 3, 4]


def test_on_a_tie_the_earlier_model_column_gets_the_lower_ranking():
    """Criterion 2: a tie is broken by column order, the earlier class ranks first (as the step does)."""
    rows = _rows_of_head(_frame(7, TIED_PROBS))  # 6 rows
    ranking_of = {r["class_id"]: r["ranking"] for r in rows}  # class_id == column position (TAX_MAPS)
    assert ranking_of[0] == 1 and ranking_of[1] == 2  # top tie: column 0 before column 1
    assert ranking_of[2] == 3  # untied class keeps its place
    assert ranking_of[3] == 4 and ranking_of[4] == 5  # middle tie: column 3 before column 4
    assert ranking_of[5] == 6  # the lowest, untied class is last


def test_without_ties_rankings_are_unchanged():
    """Criterion 3: strictly decreasing probabilities rank 1..N in column order, as the dense method did."""
    rows = _rows_of_head(_decreasing(7))  # no ties
    assert [r["ranking"] for r in sorted(rows, key=lambda r: r["class_id"])] == [1, 2, 3, 4, 5, 6]  # same as before


def test_docstring_names_the_tie_break_rule():
    """Criterion 4: the docstring states the tie-break and no longer promises a dense rank."""
    doc = pw.build_probability_rows.__doc__  # the public contract
    assert "dense" not in doc  # the old promise is gone
    assert "tie" in doc.lower()  # the new rule is named
