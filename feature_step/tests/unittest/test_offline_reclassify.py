"""Tests for features/offline/reclassify.py and the version parameter of classify.py.

The reclassification path reads stored feature rows instead of computing them,
so the pieces under test are pure: LUT name -> model column, the newest-date
filter per oid, the pivot to the model's frame, and the per-oid split of a
batched OutputDTO. The model is a fake; no DB, no pickle.
"""
import pandas as pd
import pytest

from alerce_classifiers.base.dto import OutputDTO
from features.offline import classify, reclassify


def _rows(records):
    """Rows as fetch_feature_rows returns them: oid, feature_name, band, value, updated_date."""
    return pd.DataFrame(records, columns=["oid", "feature_name", "band", "value", "updated_date"])


def test_model_column_translates_lut_spelling():
    """Hyphens and slashes become underscores; band 0 means no suffix, 12 is the (g,r) pair."""
    assert reclassify.model_column("Amplitude", 1) == "Amplitude_1"
    assert reclassify.model_column("W1-W2", 0) == "W1_W2"
    assert reclassify.model_column("g-r_mean", 12) == "g_r_mean_12"
    assert reclassify.model_column("Power_rate_1/2", 12) == "Power_rate_1_2_12"
    assert reclassify.model_column("Timespan", 0) == "Timespan"


def test_current_rows_keeps_only_the_newest_date_per_oid():
    """A row left behind by an older computation is dropped; the other oid is untouched."""
    rows = _rows([
        (1, "Amplitude", 1, 0.5, "2026-08-21"),
        (1, "Amplitude", 2, 0.6, "2026-08-21"),
        (1, "Period", 0, 9.9, "2026-08-20"),      # stale: older than its siblings
        (2, "Amplitude", 1, 0.7, "2026-08-20"),   # only date for oid 2, so it stays
    ])
    out = reclassify.current_rows(rows)
    assert sorted(zip(out["oid"], out["feature_name"])) == [
        (1, "Amplitude"), (1, "Amplitude"), (2, "Amplitude")]


def test_features_frame_pivots_to_the_model_columns():
    """One row per oid, model column names, missing features NaN, extra columns dropped."""
    rows = _rows([
        (1, "Amplitude", 1, 0.5, "d"),
        (1, "W1-W2", 0, 0.1, "d"),
        (2, "Amplitude", 1, 0.7, "d"),
        (2, "Unknown", 0, 3.0, "d"),              # not in the model: dropped
    ])
    frame = reclassify.features_frame(rows, ["Amplitude_1", "W1_W2"])
    assert list(frame.columns) == ["Amplitude_1", "W1_W2"]
    assert list(frame.index) == [1, 2]
    assert frame.loc[1, "W1_W2"] == 0.1
    assert pd.isna(frame.loc[2, "W1_W2"])


def test_features_frame_of_no_rows_is_empty_with_the_model_columns():
    frame = reclassify.features_frame(_rows([]), ["Amplitude_1"])
    assert len(frame) == 0
    assert list(frame.columns) == ["Amplitude_1"]


class _FakeModel:
    """Records the DTO it was given and answers with two oids of canned probabilities."""
    def __init__(self):
        self.received = None

    def can_predict(self, dto):
        return len(dto.features) > 0, "Empty features found"

    def predict(self, dto):
        self.received = dto
        idx = list(dto.features.index)
        return OutputDTO(
            pd.DataFrame({"AGN": [0.7, 0.2], "SNIa": [0.3, 0.8]}, index=idx),
            {"top": pd.DataFrame({"Stochastic": [0.7, 0.2], "Transient": [0.3, 0.8]}, index=idx),
             "children": {"Transient": pd.DataFrame({"SNIa": [1.0, 1.0]}, index=idx),
                          "Stochastic": pd.DataFrame({"AGN": [1.0, 1.0]}, index=idx)}},
        )


def test_classify_frame_hands_the_frame_to_the_model_as_features():
    frame = pd.DataFrame({"Amplitude_1": [0.5, 0.7]}, index=[1, 2])
    model = _FakeModel()
    dto = reclassify.classify_frame(frame, model)
    assert list(model.received.features.index) == [1, 2]
    assert model.received.detections.empty
    assert list(dto.probabilities.index) == [1, 2]


def test_classify_frame_returns_empty_output_when_the_model_cannot_predict():
    dto = reclassify.classify_frame(pd.DataFrame(columns=["Amplitude_1"]), _FakeModel())
    assert len(dto.probabilities) == 0


def test_split_output_gives_one_oid_single_row_frames():
    """build_probability_rows is per oid, so the batched DTO is sliced to one row each."""
    frame = pd.DataFrame({"Amplitude_1": [0.5, 0.7]}, index=[1, 2])
    dto = reclassify.classify_frame(frame, _FakeModel())
    one = reclassify.split_output(dto, 2)
    assert list(one.probabilities.index) == [2]
    assert one.probabilities.loc[2, "SNIa"] == 0.8
    assert list(one.hierarchical["top"].index) == [2]
    assert list(one.hierarchical["children"]["Transient"].index) == [2]


def test_split_output_of_an_absent_oid_is_empty():
    frame = pd.DataFrame({"Amplitude_1": [0.5, 0.7]}, index=[1, 2])
    dto = reclassify.classify_frame(frame, _FakeModel())
    assert len(reclassify.split_output(dto, 99).probabilities) == 0


def test_resolve_model_version_accepts_the_expected_version():
    """A 2.2.0 pickle is accepted when the caller says 2.2.0; the pin stays the default."""
    assert classify.resolve_model_version("2.2.0", expected="2.2.0") == "2.2.0"
    assert classify.resolve_model_version("no_version", expected="2.2.0") == "2.2.0"
    assert classify.resolve_model_version("2.1.0") == "2.1.0"


def test_resolve_model_version_refuses_a_mismatch():
    with pytest.raises(ValueError, match="2.2.0"):
        classify.resolve_model_version("2.1.0", expected="2.2.0")
    with pytest.raises(ValueError, match="2.1.0"):
        classify.resolve_model_version("2.2.0")


def test_current_rows_keeps_rows_whose_oid_has_no_date_at_all():
    """A NULL updated_date cannot be compared: rows survive when the oid has no dated row."""
    rows = _rows([
        (1, "Amplitude", 1, 0.5, None),            # oid 1: both rows undated -> kept
        (1, "Amplitude", 2, 0.6, None),
        (2, "Amplitude", 1, 0.7, "2026-08-21"),    # oid 2: an undated row next to a dated one
        (2, "Period", 0, 9.9, None),               #        is of unknown age -> dropped
    ])
    out = reclassify.current_rows(rows)
    assert sorted(zip(out["oid"], out["feature_name"])) == [
        (1, "Amplitude"), (1, "Amplitude"), (2, "Amplitude")]


def test_features_frame_refuses_two_rows_on_the_same_model_column():
    """Two LUT names collapsing onto one column would silently pick one value; refuse instead."""
    rows = _rows([
        (1, "W1-W2", 0, 0.1, "d"),
        (1, "W1_W2", 0, 0.2, "d"),                 # a second spelling of the same column
    ])
    with pytest.raises(ValueError):
        reclassify.features_frame(rows, ["W1_W2"])
