from alerce_classifiers.rf_features_header_classifier.mapper import BarneyMapper
from alerce_classifiers.rf_features_header_classifier.utils import FEAT_DICT
from tests import utils
from tests.mockdata.mock_message import MESSAGES

import pandas as pd

feat_dict = FEAT_DICT

mock_input_dto = utils.create_input_dto(MESSAGES)


def check_correct_input(input: pd.DataFrame):
    columns = list(mock_input_dto.features.columns) + list(feat_dict.values())
    assert all(col in input.columns for col in columns)


def test_preprocess():
    mock_input_dto = utils.create_input_dto(MESSAGES)
    mapper = BarneyMapper()

    preprocessed_input = mapper.preprocess(mock_input_dto)

    check_correct_input(preprocessed_input)


def test_postprocess():
    mapper = BarneyMapper()
    probabilities = pd.DataFrame({"aid": ["aid1"], "SN": [1]})
    dto = mapper.postprocess(probabilities)
    assert dto.probabilities.aid.iloc[0] == probabilities.aid.iloc[0]
