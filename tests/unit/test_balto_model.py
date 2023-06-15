from alerce_classifiers.balto.mapper import BaltoMapper
from alerce_classifiers.balto.model import BaltoClassifier
from alerce_classifiers.balto.utils import FEAT_DICT
from tests import utils
from unittest.mock import patch


def check_model_quantiles(model: BaltoClassifier):
    assert len(model.quantiles) == len(FEAT_DICT.values())


@patch("alerce_classifiers.balto.model.torch")
@patch("alerce_classifiers.balto.model.load")
def test_constructor(_, __):
    mapper = BaltoMapper()
    model = BaltoClassifier(
        model_path="",
        header_quantiles_path="",
        mapper=mapper,
    )
    check_model_quantiles(model)


@patch("alerce_classifiers.balto.model.torch")
def test_predict(torch):
    input_dto = utils.create_mock_dto()
    model = BaltoClassifier(
        "model_path",
        "https://assets.alerce.online/pipeline/elasticc/balto_classifier_lc_header/3.0.0/HeaderNorm1QT/",
        BaltoMapper(),
    )
    output_dto = model.predict(input_dto)
    model.model.predict_mix.assert_called()
    assert len(output_dto.probabilities) == len(input_dto.detections)
