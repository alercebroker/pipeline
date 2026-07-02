"""Tests for the offline 199-feature verification (model_feature_list + model_features)."""
import pandas as pd
from features.offline.model_feature_list import MODEL_FEATURE_LIST, MODEL_VERSION, MODEL_MD5
from features.offline import model_features


def test_model_feature_list_has_199_unique_names():
    assert len(MODEL_FEATURE_LIST) == 199
    assert len(set(MODEL_FEATURE_LIST)) == 199


def test_model_provenance_pins_deployed_artifact():
    assert MODEL_VERSION == "2.1.0"
    assert MODEL_MD5 == "95e8e9f18fde62f22025e31a88ad81fa"


def test_diff_feature_coverage_reports_missing_and_extra():
    produced = ["Amplitude_1", "Amplitude_2", "surprise_1"]
    expected = ["Amplitude_1", "Amplitude_2", "Std_1"]
    diff = model_features.diff_feature_coverage(produced, expected)
    assert diff["missing"] == ["Std_1"]      # would KeyError at predict
    assert diff["extra"] == ["surprise_1"]
    assert diff["covered"] == ["Amplitude_1", "Amplitude_2"]
    assert diff["n_expected"] == 3
    assert diff["n_missing"] == 1


def test_predict_input_columns_matches_predict_path(monkeypatch):
    # parse_output is monkeypatched (mirrors test_offline_classify.py) so no real
    # AstroObject/DB is needed. The band suffix + None values exercise the exact
    # SquidwardMapper.preprocess (None->NaN) + RandomForestPreprocessor path.
    out_message = {
        "oid": 123,
        "features": {"Amplitude_1": 0.5, "W1_W2": None, "ps_g_r": 1.0},
    }
    monkeypatch.setattr(model_features, "parse_output",
                        lambda aos, msgs, candids: [out_message])

    cols = model_features.predict_input_columns(object(), {"oid": 123, "measurement_id": [1]})

    # RandomForestPreprocessor is idempotent on these already-suffixed names.
    assert set(cols) == {"Amplitude_1", "W1_W2", "ps_g_r"}
