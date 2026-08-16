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
