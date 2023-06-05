from unittest.mock import patch
from alerce_classifiers.base.factories import input_dto_factory

from alerce_classifiers.rf_features_header_classifier.model import (
    RandomForestFeaturesHeaderClassifier,
)
import pytest


@patch("alerce_classifiers.rf_features_header_classifier.model.ElasticcRandomForest")
def test_init(elasticc_random_forest):
    RandomForestFeaturesHeaderClassifier("cualquier_cosa")
    elasticc_random_forest.assert_called()


@patch("alerce_classifiers.rf_features_header_classifier.model.ElasticcRandomForest")
def test_predict(elasticc_random_forest):
    model = RandomForestFeaturesHeaderClassifier("cualquier_cosa")
    # TODO: create input dto with features and detections + extra_fields
    input_dto = input_dto_factory()
    output_dto = model.predict(input_dto)
    assert output_dto.probabilities.loc["aid1"].sum() == pytest.approx(1)
