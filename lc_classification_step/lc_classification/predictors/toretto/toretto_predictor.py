from alerce_classifiers.base.dto import InputDTO
from alerce_classifiers.rf_features_classifier.model import (
    RandomForestFeaturesClassifier,
)

from lc_classification.predictors.predictor.predictor_parser import PredictorInput
from ..predictor.predictor import Predictor


class TorettoPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = str(kwargs["model_path"])
        if not self.model:
            self.model = RandomForestFeaturesClassifier(model_path)

    def _predict(self, model_input: InputDTO):
        return self.model.predict(model_input)

    def get_feature_list(self):
        return self.model.feature_list
