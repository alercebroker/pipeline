"""Tests for the offline 199-feature verification (model_feature_list + model_features)."""
from features.offline.model_feature_list import MODEL_FEATURE_LIST, MODEL_VERSION, MODEL_MD5


def test_model_feature_list_has_199_unique_names():
    assert len(MODEL_FEATURE_LIST) == 199
    assert len(set(MODEL_FEATURE_LIST)) == 199


def test_model_provenance_pins_deployed_artifact():
    assert MODEL_VERSION == "2.1.0"
    assert MODEL_MD5 == "95e8e9f18fde62f22025e31a88ad81fa"
