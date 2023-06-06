from alerce_classifiers.base.factories import input_dto_factory
from alerce_classifiers.rf_features_header_classifier.mapper import (
    RandomForestClassifierMapper as BarneyMapper,
)
from alerce_classifiers.utils.input_mapper.elasticc.dict_transform import FEAT_DICT
from fastavro import utils
from mockdata.input import DETECTIONS
from mockdata.input import FEATURES

import pandas as pd

feat_dict = FEAT_DICT

mock_detections = pd.DataFrame(utils.generate_one(DETECTIONS))
mock_features = pd.DataFrame(utils.generate_one(FEATURES))


def check_header_correct(headers):
    assert all(header in feat_dict.values() for header in headers.columns)


def check_features_correct(features):
    pd.testing.assert_frame_equal(features, mock_features)


def test_preprocess():
    dto = input_dto_factory(mock_detections, None, mock_features, None, None)
    mapper = BarneyMapper()

    preprocessed_input = mapper.preprocess(dto)

    check_header_correct(preprocessed_input[0])
    check_features_correct(preprocessed_input[1])


def test_postprocess():
    mapper = BarneyMapper()
    probabilities = pd.DataFrame({"aid": ["aid1"], "SN": [1]})
    dto = mapper.postprocess(probabilities)
    assert dto.probabilities.aid.iloc[0] == probabilities.aid.iloc[0]
