from alerce_classifiers.base.factories import input_dto_factory
from alerce_classifiers.rf_features_header_classifier.mapper import BarneyMapper
from alerce_classifiers.rf_features_header_classifier.utils import FEAT_DICT
from mockdata.detections import DETECTIONS
from mockdata.features import FEATURES

import pandas as pd

feat_dict = FEAT_DICT

mock_detections = pd.DataFrame(DETECTIONS)
mock_features = pd.DataFrame(FEATURES)


def check_correct_input(input: pd.DataFrame):
    columns = list(mock_features.columns) + list(feat_dict.values())
    assert all(col in input.columns for col in columns)


def test_preprocess():
    dto = input_dto_factory(mock_detections, None, mock_features, None, None)
    mapper = BarneyMapper()

    preprocessed_input = mapper.preprocess(dto)

    check_correct_input(preprocessed_input)


def test_postprocess():
    mapper = BarneyMapper()
    probabilities = pd.DataFrame({"aid": ["aid1"], "SN": [1]})
    dto = mapper.postprocess(probabilities)
    assert dto.probabilities.aid.iloc[0] == probabilities.aid.iloc[0]
