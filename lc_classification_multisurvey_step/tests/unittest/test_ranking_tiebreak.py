"""Ranking is a permutation of 1..N per (oid, head); ties break by column order.

One test per acceptance criterion of
docs/superpowers/specs/2026-10-22-lc-ms-ranking-tiebreak-design.md.
Reuses the stub DTO, taxonomy and `build` helper of test_probabilities.py.
"""
import itertools  # permutations of rows for the pairwise rank check

from tests.unittest.test_probabilities import (  # the helpers of the sibling file, not duplicated here
    TAXONOMY,
    build,
    frame,
    make_dto,
)


def ranks_by_class(rows, classifier_id):
    """{class_id: ranking} for the rows of one head."""
    return {r["class_id"]: r["ranking"] for r in rows if r["classifier_id"] == classifier_id}  # one head only


class TestRankingTiebreak:
    """Ranking of `build_probability_rows` on tied probabilities, one test per spec criterion."""

    def test_rankings_are_a_permutation_of_1_to_n_even_with_ties(self):
        """Criterion 1: no repeated rank and no gap when probabilities tie."""
        dto = make_dto(flat=frame([1, 2], {"SNIa": [0.5, 0.2], "AGN": [0.5, 0.2], "LPV": [0.0, 0.6]}))  # two oids, both tied

        rows = build(dto, {1: 1.0, 2: 1.0})  # flat head only, 3 rows per oid

        for oid in (1, 2):  # every (oid, head) group independently
            rankings = sorted(r["ranking"] for r in rows if r["oid"] == oid)  # the head has 3 classes
            assert rankings == [1, 2, 3]  # exactly {1..N}, no repeat, no gap

    def test_larger_probability_always_gets_smaller_rank(self):
        """Criterion 2: a strictly larger probability never ranks below a smaller one."""
        dto = make_dto(flat=frame([1], {"SNIa": [0.2], "AGN": [0.7], "LPV": [0.1]}))  # unsorted columns

        rows = build(dto, {1: 1.0})  # three rows, distinct probabilities

        for a, b in itertools.permutations(rows, 2):  # every ordered pair of rows of the oid
            if a["probability"] > b["probability"]:  # strictly larger probability ...
                assert a["ranking"] < b["ranking"]  # ... must have the strictly smaller rank

    def test_tie_goes_to_the_first_column_of_the_frame(self):
        """Criterion 3: on a tie the class that comes first in the model's columns wins."""
        snia, agn, lpv = TAXONOMY[50]["SNIa"], TAXONOMY[50]["AGN"], TAXONOMY[50]["LPV"]  # class ids of the flat head

        rows = build(make_dto(flat=frame([1], {"SNIa": [0.5], "AGN": [0.5], "LPV": [0.0]})), {1: 1.0})  # SNIa column first
        assert ranks_by_class(rows, 50) == {snia: 1, agn: 2, lpv: 3}  # SNIa wins the tie

        rows = build(make_dto(flat=frame([1], {"AGN": [0.5], "SNIa": [0.5], "LPV": [0.0]})), {1: 1.0})  # AGN column first
        assert ranks_by_class(rows, 50) == {agn: 1, snia: 2, lpv: 3}  # AGN wins the same tie

    def test_tiebreak_holds_per_head_in_a_multi_head_dto(self):
        """Criterion 1 and 3 across heads: each head ranks its own columns independently."""
        dto = make_dto(  # two heads, each with its own tie
            flat=frame([1], {"SNIa": [0.4], "AGN": [0.4], "LPV": [0.2]}),  # tie in the flat head
            periodic=frame([1], {"EA": [0.5], "LPV": [0.5]}),  # tie in the periodic head, EA first
        )  # top, transient and stochastic heads are absent and skipped

        rows = build(dto, {1: 1.0})  # 3 flat rows + 2 periodic rows

        assert ranks_by_class(rows, 50) == {TAXONOMY[50]["SNIa"]: 1, TAXONOMY[50]["AGN"]: 2, TAXONOMY[50]["LPV"]: 3}  # flat head
        assert ranks_by_class(rows, 90) == {TAXONOMY[90]["EA"]: 1, TAXONOMY[90]["LPV"]: 2}  # periodic head, EA first
