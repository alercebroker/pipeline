from alerce_classifiers.rf_features_classifier.model import (
    RandomForestFeaturesClassifier,
)
from pandas import DataFrame

from lc_classification.predictors.predictor.predictor_parser import PredictorInput
from ..predictor.predictor import Predictor


class ElasticcRandomForestPredictor(Predictor):
    def __init__(self, model=None, **kwargs):
        model_path = str(kwargs["model_path"])
        self.model = model or RandomForestFeaturesClassifier(model_path)

    def predict(self, model_input: PredictorInput[DataFrame]):
        return self.model.predict(model_input.value)

    def get_feature_list(self):
        return self.model.feature_list
