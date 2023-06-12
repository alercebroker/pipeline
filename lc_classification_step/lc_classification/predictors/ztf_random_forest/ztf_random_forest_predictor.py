from lc_classifier.classifier.models import HierarchicalRandomForest
from pandas import DataFrame
from lc_classification.predictors.predictor.predictor import Predictor
from lc_classification.predictors.predictor.predictor_parser import PredictorInput


class ZtfRandomForestPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.model:
            self.model = HierarchicalRandomForest({})
        self.model.download_model()
        self.model.load_model(self.model.MODEL_PICKLE_PATH)

    def _predict(self, model_input: PredictorInput[DataFrame]):
        model_result = self.model.predict_in_pipeline(model_input.value)
        return model_result

    def can_predict(self, model_input: PredictorInput[DataFrame]):
        return model_input.value.any().any()

    def get_feature_list(self):
        return self.model.feature_list
