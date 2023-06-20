from alerce_classifiers.base.dto import InputDTO
from alerce_classifiers.rf_features_header_classifier.model import (
    RandomForestFeaturesHeaderClassifier,
)
from alerce_classifiers.rf_features_header_classifier.utils import FEAT_DICT

from lc_classification.predictors.predictor.predictor_parser import PredictorInput

from ..predictor.predictor import Predictor


class BarneyPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = str(kwargs["model_path"])
        if not self.model:
            self.model = RandomForestFeaturesHeaderClassifier(model_path)

    def _predict(self, model_input: PredictorInput[InputDTO]):
        return self.model.predict(model_input.value)

    def get_feature_list(self):
        return FEAT_DICT.values()
