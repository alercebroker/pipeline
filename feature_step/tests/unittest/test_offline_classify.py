"""Tests for features/offline/classify.py — the message->DTO->prediction bridge.

The real SquidwardFeaturesClassifier is replaced by fakes; parse_output is
monkeypatched so no real AstroObject/DB is needed. Requires alerce_classifiers
installed (for OutputDTO / input_dto_factory)."""
import pandas as pd
import pytest

from alerce_classifiers.base.dto import OutputDTO
from features.offline import classify


def test_features_message_to_dto_builds_wide_frame():
    out_message = {"oid": 123, "features": {"Amplitude_1": 0.5, "Period_2": 2.0}}
    dto = classify.features_message_to_dto(out_message)
    feats = dto.features
    assert list(feats.index) == [123]
    assert feats.loc[123, "Amplitude_1"] == 0.5
    assert feats.loc[123, "Period_2"] == 2.0
    assert dto.detections.empty


def test_features_message_to_dto_handles_missing_features():
    out_message = {"oid": 9, "features": None}
    dto = classify.features_message_to_dto(out_message)
    assert list(dto.features.index) == [9]


class _FakeModel:
    """Stub model: records the DTO it received and returns canned probabilities."""
    def __init__(self):
        self.received = None

    def can_predict(self, dto):
        return True, ""

    def predict(self, dto):
        self.received = dto
        return OutputDTO(
            pd.DataFrame({"AGN": [0.7]}, index=[123]),
            {"top": pd.DataFrame(), "children": {}},
        )


class _CantModel:
    def can_predict(self, dto):
        return False, "Empty features found"

    def predict(self, dto):
        raise AssertionError("predict must not be called when can_predict is False")


def test_classify_astro_object_predicts(monkeypatch):
    out_message = {"oid": 123, "measurement_id": [1], "features": {"Amplitude_1": 0.5}}
    monkeypatch.setattr(classify, "parse_output", lambda aos, msgs, candids: [out_message])
    model = _FakeModel()

    result = classify.classify_astro_object(object(), {"oid": 123, "measurement_id": [1]}, model)

    assert result.probabilities.loc[123, "AGN"] == 0.7
    assert model.received.features.loc[123, "Amplitude_1"] == 0.5


def test_classify_astro_object_cant_predict_returns_empty(monkeypatch):
    out_message = {"oid": 1, "measurement_id": [], "features": {}}
    monkeypatch.setattr(classify, "parse_output", lambda aos, msgs, candids: [out_message])

    result = classify.classify_astro_object(object(), {"oid": 1, "measurement_id": []}, _CantModel())

    assert result.probabilities.empty


def test_load_squidward_model_requires_model_path(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with pytest.raises(ValueError, match="MODEL_PATH"):
        classify.load_squidward_model()
