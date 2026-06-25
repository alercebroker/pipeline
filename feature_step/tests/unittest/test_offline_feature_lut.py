"""Unit tests for the offline ZTF feature LUT fixture + loaders.

Asserts structural invariants that hold for any valid generated fixture, so
the tests don't hard-code the feature-name list.
"""
from features.offline.feature_lut import (
    FEATURE_NAME_LUT,
    FEATURE_VERSION_LUT,
    load_feature_name_lut,
    version_name_to_id,
)


def test_name_lut_non_empty():
    assert len(FEATURE_NAME_LUT) > 0


def test_name_lut_ids_contiguous_from_zero():
    assert sorted(FEATURE_NAME_LUT) == list(range(len(FEATURE_NAME_LUT)))


def test_name_lut_sorted_by_name():
    names = [FEATURE_NAME_LUT[i] for i in sorted(FEATURE_NAME_LUT)]
    assert names == sorted(names)


def test_load_returns_independent_copy():
    lut = load_feature_name_lut()
    assert lut == FEATURE_NAME_LUT
    lut[10_000] = "x"
    assert 10_000 not in FEATURE_NAME_LUT


def test_version_round_trips():
    vid, vname = next(iter(FEATURE_VERSION_LUT.items()))
    assert version_name_to_id(vname) == vid


def test_unknown_version_warns_and_returns_negative_one():
    assert version_name_to_id("__definitely_not_a_version__") == -1
