from alerce_classifiers.base.factories import input_dto_factory
from alerce_classifiers.rf_features_header_classifier.model import (
    RandomForestFeaturesHeaderClassifier,
)
from mockdata.detections import DETECTIONS
from mockdata.features import FEATURES
from unittest.mock import patch

import pandas as pd

mock_detections = pd.DataFrame(DETECTIONS)
mock_features = pd.DataFrame(FEATURES)
mock_probabilities = pd.DataFrame(
    {"aid": ["aid1"], "SN": [0.5], "AGN": [0.4], "Other": [0.3]}
)


@patch("alerce_classifiers.rf_features_header_classifier.model.ElasticcRandomForest")
def test_init(elasticc_random_forest):
    RandomForestFeaturesHeaderClassifier("cualquier_cosa")
    elasticc_random_forest.assert_called()


@patch("alerce_classifiers.rf_features_header_classifier.model.ElasticcRandomForest")
def test_predict(elasticc_random_forest):
    elasticc_random_forest.return_value.predict_proba.return_value = mock_probabilities

    model = RandomForestFeaturesHeaderClassifier("cualquier_cosa")

    input_dto = input_dto_factory(mock_detections, None, mock_features, None, None)
    output_dto = model.predict(input_dto)

    pd.testing.assert_frame_equal(output_dto.probabilities, mock_probabilities)
