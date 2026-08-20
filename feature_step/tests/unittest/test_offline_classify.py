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


def test_classify_oid_for_save_returns_lastmjd(monkeypatch):
    import pandas as pd
    # Fake the DB readers + message/AO build so no real DB is needed.
    monkeypatch.setattr(classify.db, "fetch_detections",
                        lambda c, oids: pd.DataFrame({"mjd": [59000.0, 59010.5]}))
    monkeypatch.setattr(classify.db, "fetch_forced_photometry",
                        lambda c, oids: pd.DataFrame({"mjd": [59020.25]}))  # forced later than dets
    monkeypatch.setattr(classify.db, "fetch_ps1", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify.db, "fetch_allwise", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify.db, "fetch_references", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify, "build_message", lambda oid, d, f, p: {"oid": oid})
    monkeypatch.setattr(classify, "compute_astro_object",
                        lambda *a, **k: object())  # non-None AO
    monkeypatch.setattr(classify, "classify_astro_object",
                        lambda ao, msg, model: OutputDTO(pd.DataFrame({"AGN": [0.9]}, index=[123]),
                                                         {"top": pd.DataFrame(), "children": {}}))

    dto, lastmjd, matches = classify.classify_oid_for_save(123, "creds", model=object())
    assert lastmjd == 59020.25            # max over detections + forced, already MJD
    assert dto.probabilities.loc[123, "AGN"] == 0.9
    assert matches == []                  # no xmatch_url -> DB fallback, no live matches


def test_classify_oid_for_save_none_when_no_ao(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(classify.db, "fetch_detections", lambda c, oids: pd.DataFrame({"mjd": []}))
    monkeypatch.setattr(classify.db, "fetch_forced_photometry", lambda c, oids: pd.DataFrame({"mjd": []}))
    monkeypatch.setattr(classify.db, "fetch_ps1", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify.db, "fetch_allwise", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify.db, "fetch_references", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify, "build_message", lambda oid, d, f, p: {"oid": oid})
    monkeypatch.setattr(classify, "compute_astro_object", lambda *a, **k: None)  # too few dets

    dto, lastmjd, matches = classify.classify_oid_for_save(1, "creds", model=object())
    assert dto is None and lastmjd is None and matches == []


def test_resolve_model_version_falls_back_to_the_pinned_version():
    """The version must come from the code, not from the shape of a file path.

    alerce_classifiers derives it by scanning MODEL_PATH's components for
    something version-shaped, so a local pickle at /data/models/model.pkl yields
    the literal "no_version". Requiring operators to bury the file under a
    2.1.0/ directory to make a version appear is a filesystem convention
    standing in for a constant we already pin.
    """
    from features.offline.model_feature_list import MODEL_VERSION
    assert classify.resolve_model_version("no_version") == MODEL_VERSION


def test_resolve_model_version_keeps_a_real_reported_version():
    assert classify.resolve_model_version("2.1.0") == "2.1.0"


def test_resolve_model_version_refuses_a_version_that_is_not_the_pinned_one():
    """A path saying 9.9.9 means the artifact is not the one we validated.

    MODEL_FEATURE_LIST, the seeded taxonomy and CLASSIFIER_VERSION are all
    pinned to 2.1.0, so silently classifying with a different model would write
    probabilities whose class ids mean something else.
    """
    with pytest.raises(ValueError, match="9.9.9"):
        classify.resolve_model_version("9.9.9")
