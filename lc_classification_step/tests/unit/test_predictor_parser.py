from lc_classification.predictors.predictor.predictor_parser import PredictorOutput
import pytest


def test_predictor_output_valid():
    po = PredictorOutput({"probabilities": "test"})
    assert po.classifications is not None


def test_predictor_output_invalid():
    with pytest.raises(AssertionError):
        PredictorOutput({"invalid": True})
