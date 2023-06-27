from alerce_classifiers.base.dto import InputDTO
from alerce_classifiers.transformer_lc_features.mapper import LCFeatureMapper
from alerce_classifiers.transformer_lc_features.model import (
    TransformerLCFeaturesClassifier,
)
from alerce_classifiers.transformer_lc_features.utils import FEATURES_ORDER

from lc_classification.predictors.predictor.predictor import Predictor
from lc_classification.predictors.predictor.predictor_parser import PredictorInput


class MessiPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = str(kwargs["model_path"])
        header_quantiles_path = str(kwargs["header_quantiles_path"])
        feature_quantiles_path = str(kwargs["feature_quantiles_path"])
        if not self.model:
            self.model = TransformerLCFeaturesClassifier(
                model_path,
                header_quantiles_path,
                feature_quantiles_path,
                mapper=LCFeatureMapper(),
            )

    def _predict(self, model_input: PredictorInput[InputDTO]):
        return self.model.predict(model_input.value)

    def get_feature_list(self):
        return FEATURES_ORDER
