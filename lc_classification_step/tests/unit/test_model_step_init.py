from unittest import mock, TestCase
from lc_classification.core.step import LateClassifier
from .conftest import ztf_config, balto_config, barney_config, toretto_config, messi_config


class ModelInitTestCase(TestCase):
    @mock.patch("lc_classification.core.step.get_class")
    def test_ztf_init(self, mock_get_class):
        mock_model_class = mock.MagicMock()
        mock_model_class.return_value = "HierarchicalRandomForest"
        mock_get_class.return_value = mock_model_class
        ztf_step = LateClassifier(config = ztf_config)

        mock_get_class.assert_called_once_with(ztf_config["MODEL_CONFIG"]["CLASS"])      
        mock_model_class.return_value.download_model.assert_called_once()
        mock_model_class.return_value.load_model.assert_called_once_with(
            mock_model_class.return_value.MODEL_PICKLE_PATH
        )

    @mock.patch("lc_classification.core.step.get_class")
    def test_balto_init(self, mock_get_class):
        mock_model_class = mock.MagicMock()
        mock_model_class.return_value = "BaltoClassifier"
        mock_get_class.return_value = mock_model_class
        balto_step = LateClassifier(config = balto_config)

        mock_get_class.assert_called_once_with(balto_config["MODEL_CONFIG"]["CLASS"])

    @mock.patch("lc_classification.core.step.get_class")
    def test_barney_init(self, mock_get_class):
        mock_model_class = mock.MagicMock()
        mock_model_class.return_value = "RandomForestFeaturesHeaderClassifier"
        mock_get_class.return_value = mock_model_class
        barney_step = LateClassifier(config = barney_config)

        mock_get_class.assert_called_once_with(barney_config["MODEL_CONFIG"]["CLASS"])

        # correr el init del step

        # asser que get class se llamo con los datos correctos
        # assert que model class se llamo con los datos correctos
    @mock.patch("lc_classification.core.step.get_class")
    def test_messi_init(self, mock_get_class):
        mock_model_class = mock.MagicMock()
        mock_model_class.return_value = "MessiClassifier"
        mock_get_class.return_value = mock_model_class
        messi_step = LateClassifier(config = messi_config)

        mock_get_class.assert_called_once_with(messi_config["MODEL_CONFIG"]["CLASS"])

        # correr el init del step

        # asser que get class se llamo con los datos correctos
        # assert que model class se llamo con los datos correctos
    @mock.patch("lc_classification.core.step.get_class")
    def test_toretto_init(self, mock_get_class):
        mock_model_class = mock.MagicMock()
        mock_model_class.return_value = "RandomForestFeaturesClassifier"
        mock_get_class.return_value = mock_model_class
        toretto_step = LateClassifier(config = toretto_config)

        mock_get_class.assert_called_once_with(toretto_config["MODEL_CONFIG"]["CLASS"])

        # correr el init del step

        # asser que get class se llamo con los datos correctos
        # assert que model class se llamo con los datos correctos
