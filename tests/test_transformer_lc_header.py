from alerce_classifiers.transformer_lc_header.dict_transform import FEAT_DICT
from alerce_classifiers.transformer_lc_header.model import TransformerLCHeaderClassifier
from alerce_classifiers.transformer_lc_header.mapper import LCHeaderMapper
from unittest.mock import patch
from alerce_classifiers.base.factories import input_dto_factory
import pandas as pd

from tests.mockdata.detections import DETECTIONS


def check_model_quantiles(model: TransformerLCHeaderClassifier):
    assert len(model.quantiles) == len(FEAT_DICT.values())


@patch("alerce_classifiers.transformer_lc_header.model.torch")
@patch("alerce_classifiers.transformer_lc_header.model.load")
def test_constructor(_, __):
    mapper = LCHeaderMapper()
    model = TransformerLCHeaderClassifier(
        model_path="",
        header_quantiles_path="",
        mapper=mapper,
    )
    check_model_quantiles(model)


@patch("alerce_classifiers.transformer_lc_header.model.torch")
def test_predict(torch):
    mock_detections = pd.DataFrame(DETECTIONS)
    input_dto = input_dto_factory(
        detections=mock_detections,
        non_detections=pd.DataFrame(),
        features=pd.DataFrame(),
        xmatch=pd.DataFrame(),
        stamps=pd.DataFrame(),
    )
    model = TransformerLCHeaderClassifier(
        "model_path",
        "https://assets.alerce.online/pipeline/elasticc/balto_classifier_lc_header/3.0.0/HeaderNorm1QT/",
        LCHeaderMapper(),
    )
    output_dto = model.predict(input_dto)
    model.model.predict_mix.assert_called()
    assert len(output_dto.probabilities) == len(mock_detections)
