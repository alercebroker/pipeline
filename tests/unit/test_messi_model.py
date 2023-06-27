from alerce_classifiers.messi.mapper import MessiMapper
from alerce_classifiers.messi.model import MessiClassifier
from alerce_classifiers.messi.utils import FEATURES_ORDER
from tests import utils
from unittest.mock import MagicMock
from unittest.mock import patch


def check_model_quantiles(model: MessiClassifier):
    assert len(model.feature_quantiles) == len(FEATURES_ORDER)


@patch("alerce_classifiers.messi.model.BaltoClassifier")
@patch("alerce_classifiers.messi.model.torch")
@patch("alerce_classifiers.messi.model.load")
def test_constructor(_, __, ___):
    model = MessiClassifier(
        model_path="",
        header_quantiles_path="",
        feature_quantiles_path="",
        mapper=MessiMapper(),
    )
    check_model_quantiles(model)


@patch("alerce_classifiers.messi.model.BaltoClassifier")
@patch("alerce_classifiers.messi.model.torch")
def test_predict(torch, _):
    input_dto = utils.create_mock_dto()

    mapper = MessiMapper()
    mapper.preprocess = MagicMock(return_value=(MagicMock(), MagicMock()))

    model = MessiClassifier(
        "model_path",
        "https://assets.alerce.online/pipeline/elasticc/messi/1.0.0/HeaderNorm1QT/",
        "https://assets.alerce.online/pipeline/elasticc/messi/1.0.0/FeatNorm1QT/",
        mapper,
    )

    model.predict(input_dto)

    model.model.predict_mix.assert_called()
