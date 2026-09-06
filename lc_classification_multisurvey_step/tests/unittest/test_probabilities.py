"""Pure-function tests for the probability row builder.

Deliberately imports no alerce_classifiers: `probabilities.py` is duck-typed over
anything with `.probabilities` and `.hierarchical`, so a namespace stub stands in
for the real OutputDTO.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from lc_classification_multisurvey_step import probabilities as p


def make_dto(flat=None, top=None, transient=None, stochastic=None, periodic=None):
    """Stand-in for alerce_classifiers OutputDTO (probabilities + hierarchical)."""
    children = {}
    if transient is not None:
        children["Transient"] = transient
    if stochastic is not None:
        children["Stochastic"] = stochastic
    if periodic is not None:
        children["Periodic"] = periodic
    return SimpleNamespace(
        probabilities=flat if flat is not None else pd.DataFrame(),
        hierarchical={"top": top, "children": children},
    )


def frame(index, data):
    """{class_name: [values]} -> DataFrame indexed by oid, like the model emits."""
    df = pd.DataFrame(data, index=index)
    df.index.name = "oid"
    return df


class TestHeadNames:
    def test_default_base_name_matches_the_seeded_classifier(self):
        assert p.DEFAULT_CLASSIFIER_NAME == "lc_classifier_BHRF_forced_phot"

    def test_five_heads_in_flat_top_transient_stochastic_periodic_order(self):
        assert p.head_names("base") == [
            "base",
            "base_top",
            "base_transient",
            "base_stochastic",
            "base_periodic",
        ]

    def test_defaults_to_the_deployed_base_name(self):
        assert p.head_names()[0] == "lc_classifier_BHRF_forced_phot"
        assert p.head_names()[4] == "lc_classifier_BHRF_forced_phot_periodic"


class TestClassifierVersionToSmallint:
    def test_three_part_version(self):
        assert p.classifier_version_to_smallint("2.1.0") == 210

    def test_strips_suffix_on_the_patch_part(self):
        assert p.classifier_version_to_smallint("2.1.0_rc1") == 210

    def test_non_three_part_version_is_zero(self):
        assert p.classifier_version_to_smallint("dev") == 0
        assert p.classifier_version_to_smallint("2.1") == 0


class TestIterHeadFrames:
    def test_pairs_each_head_name_with_its_frame(self):
        flat = frame([1], {"SNIa": [0.9]})
        top = frame([1], {"Transient": [0.8]})
        transient = frame([1], {"SNIa": [0.7]})
        stochastic = frame([1], {"AGN": [0.6]})
        periodic = frame([1], {"LPV": [0.5]})
        dto = make_dto(flat, top, transient, stochastic, periodic)

        got = p.iter_head_frames(dto, "base")

        assert [name for name, _ in got] == p.head_names("base")
        assert got[0][1] is flat
        assert got[1][1] is top
        assert got[2][1] is transient
        assert got[3][1] is stochastic
        assert got[4][1] is periodic

    def test_missing_children_yield_none_rather_than_raising(self):
        dto = make_dto(flat=frame([1], {"SNIa": [0.9]}), top=None)
        got = dict(p.iter_head_frames(dto, "base"))
        assert got["base_top"] is None
        assert got["base_transient"] is None

    def test_absent_hierarchical_yields_none_for_the_four_hierarchical_heads(self):
        dto = SimpleNamespace(probabilities=frame([1], {"SNIa": [0.9]}), hierarchical=None)
        got = dict(p.iter_head_frames(dto, "base"))
        assert got["base"] is not None
        assert all(got[n] is None for n in p.head_names("base")[1:])


# --- build_probability_rows ------------------------------------------------

NAMES = p.head_names("base")
IDS = {NAMES[0]: 50, NAMES[1]: 60, NAMES[2]: 70, NAMES[3]: 80, NAMES[4]: 90}
TAXONOMY = {
    50: {"SNIa": 0, "AGN": 1, "LPV": 2},
    60: {"Transient": 0, "Stochastic": 1, "Periodic": 2},
    70: {"SNIa": 0, "SLSN": 1},
    80: {"AGN": 0, "QSO": 1},
    90: {"LPV": 0, "EA": 1},
}
# Ids are deliberately NOT 5-9: a reintroduced hardcode must fail these tests.


def build(dto, lastmjd_map, ids=IDS, taxonomy=TAXONOMY, **kw):
    return p.build_probability_rows(
        dto, lastmjd_map, ids, taxonomy, base_name="base", version="2.1.0", **kw
    )


class TestBuildProbabilityRows:
    def test_empty_output_dto_yields_no_rows(self):
        assert build(make_dto(), {}) == []

    def test_none_output_dto_yields_no_rows(self):
        assert build(None, {}) == []

    def test_flat_head_row_contract(self):
        dto = make_dto(flat=frame([123], {"SNIa": [0.7], "AGN": [0.2], "LPV": [0.1]}))

        rows = build(dto, {123: 60000.5})

        assert len(rows) == 3
        by_class = {r["class_id"]: r for r in rows}
        assert by_class[0] == {
            "oid": 123,
            "sid": 0,
            "classifier_id": 50,
            "classifier_version": 210,
            "class_id": 0,
            "probability": pytest.approx(0.7),
            "ranking": 1,
            "lastmjd": 60000.5,
        }
        assert set(rows[0]) == {
            "oid", "sid", "classifier_id", "classifier_version",
            "class_id", "probability", "ranking", "lastmjd",
        }

    def test_sid_is_configurable(self):
        dto = make_dto(flat=frame([123], {"SNIa": [1.0]}))
        rows = build(dto, {123: 1.0}, sid=3)
        assert {r["sid"] for r in rows} == {3}

    def test_melts_a_multi_oid_frame(self):
        dto = make_dto(flat=frame([1, 2], {"SNIa": [0.6, 0.1], "AGN": [0.4, 0.9]}))

        rows = build(dto, {1: 100.0, 2: 200.0})

        assert len(rows) == 4
        assert {r["oid"] for r in rows} == {1, 2}
        oid2 = [r for r in rows if r["oid"] == 2]
        assert {r["lastmjd"] for r in oid2} == {200.0}
        # ranking is per (oid, head): oid 2's AGN wins even though oid 1's SNIa is higher
        agn_id = TAXONOMY[50]["AGN"]
        assert [r["ranking"] for r in oid2 if r["class_id"] == agn_id] == [1]

    def test_ranking_is_dense_descending_within_oid_and_head(self):
        dto = make_dto(flat=frame([1], {"SNIa": [0.5], "AGN": [0.5], "LPV": [0.0]}))

        rows = build(dto, {1: 1.0})

        rank_by_class = {r["class_id"]: r["ranking"] for r in rows}
        assert rank_by_class[TAXONOMY[50]["SNIa"]] == 1
        assert rank_by_class[TAXONOMY[50]["AGN"]] == 1  # tie -> same dense rank
        assert rank_by_class[TAXONOMY[50]["LPV"]] == 2  # dense, not 3

    def test_all_five_heads_are_emitted(self):
        dto = make_dto(
            flat=frame([1], {"SNIa": [1.0]}),
            top=frame([1], {"Transient": [1.0]}),
            transient=frame([1], {"SNIa": [1.0]}),
            stochastic=frame([1], {"AGN": [1.0]}),
            periodic=frame([1], {"LPV": [1.0]}),
        )

        rows = build(dto, {1: 1.0})

        assert sorted(r["classifier_id"] for r in rows) == [50, 60, 70, 80, 90]

    def test_missing_and_empty_heads_are_skipped(self):
        dto = make_dto(
            flat=frame([1], {"SNIa": [1.0]}),
            top=None,
            transient=frame([], {"SNIa": []}),
        )

        rows = build(dto, {1: 1.0})

        assert {r["classifier_id"] for r in rows} == {50}

    def test_unknown_class_raises_naming_the_head_and_the_class(self):
        dto = make_dto(
            flat=frame([1, 2], {"SNIa": [0.5, 0.5], "Nonsense": [0.5, 0.5]}),
            top=frame([1, 2], {"Transient": [1.0, 1.0]}),
        )

        with pytest.raises(ValueError, match=r"'base'.*Nonsense"):
            build(dto, {1: 1.0, 2: 2.0})

    def test_known_classes_keep_every_oid(self):
        # Counterpart to the test above: with no unknown class name nothing raises
        # and every oid is emitted.
        dto = make_dto(flat=frame([1, 2], {"SNIa": [0.5, 0.5], "AGN": [0.5, 0.5]}))

        rows = build(dto, {1: 1.0, 2: 2.0})

        assert {r["oid"] for r in rows} == {1, 2}
        assert len(rows) == 4

    def test_oid_without_lastmjd_raises_naming_it(self):
        dto = make_dto(flat=frame([1, 2], {"SNIa": [1.0, 1.0]}))

        with pytest.raises(ValueError, match=r"lastmjd.*\[2\]"):
            build(dto, {1: 1.0})

    def test_head_with_no_taxonomy_map_raises(self):
        # resolve_classifiers guarantees a map per head at startup, so a miss here
        # is a bug; plain indexing surfaces it as a KeyError on the id.
        dto = make_dto(flat=frame([1], {"SNIa": [1.0]}), top=frame([1], {"Transient": [1.0]}))
        taxonomy = {50: TAXONOMY[50]}  # no map for the top head's id

        with pytest.raises(KeyError, match="60"):
            p.build_probability_rows(
                dto, {1: 1.0}, IDS, taxonomy, base_name="base", version="2.1.0"
            )

    def test_head_with_no_resolved_id_raises(self):
        dto = make_dto(flat=frame([1], {"SNIa": [1.0]}), top=frame([1], {"Transient": [1.0]}))
        ids = {NAMES[0]: 50}  # top head never resolved

        with pytest.raises(KeyError, match=NAMES[1]):
            p.build_probability_rows(
                dto, {1: 1.0}, ids, TAXONOMY, base_name="base", version="2.1.0"
            )

    @pytest.mark.parametrize(
        "flat",
        [
            {"SNIa": [0.7], "AGN": [float("nan")], "LPV": [0.3]},  # one NaN class
            {"SNIa": [float("nan")], "AGN": [float("nan")]},  # whole row NaN
        ],
        ids=["partial", "all"],
    )
    def test_nan_probability_raises_naming_the_oid(self, flat):
        # BHRF has no path to a NaN probability (sklearn 1.4 forests take NaN
        # features natively), so one is a model fault and must stop the step.
        dto = make_dto(flat=frame([1], flat), top=frame([1], {"Transient": [1.0]}))

        with pytest.raises(ValueError, match=r"NaN.*\[1\]"):
            build(dto, {1: 100.0})

    def test_values_are_native_python_types_for_json_serialisation(self):
        import json

        dto = make_dto(flat=frame([1], {"SNIa": [0.5], "AGN": [0.5]}))
        rows = build(dto, {1: 1.0})
        json.dumps(rows)  # numpy int64/float64 would raise here

    def test_drift_in_any_head_fails_the_whole_batch(self):
        dto = make_dto(
            flat=frame([1, 2], {"SNIa": [0.9, 0.1], "AGN": [0.1, 0.9]}),
            top=frame([1, 2], {"Transient": [0.8, 0.2], "Stochastic": [0.2, 0.8]}),
            periodic=frame([1, 2], {"LPV": [0.6, 0.4], "Unseeded": [0.4, 0.6]}),
        )

        # flat and top are fine; periodic carries a class the taxonomy has not
        # seeded. Nothing is emitted for the batch — partial output would leave
        # the flat head written and the periodic head silently absent.
        with pytest.raises(ValueError, match="Unseeded"):
            build(dto, {1: 100.0, 2: 200.0})
