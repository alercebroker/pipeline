from alerce_classifiers.balto.mapper import BaltoMapper
from alerce_classifiers.balto.model import BaltoClassifier
from alerce_classifiers.balto.utils import FEAT_DICT
from alerce_classifiers.base.dto import InputDTO

from lc_classification.predictors.predictor.predictor import Predictor
from lc_classification.predictors.predictor.predictor_parser import PredictorInput


class BaltoPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = str(kwargs["model_path"])
        quantiles_path = str(kwargs["quantiles_path"])
        if not self.model:
            self.model = BaltoClassifier(
                model_path, quantiles_path, mapper=BaltoMapper()
            )

    def _predict(self, model_input: InputDTO):
        return self.model.predict(model_input)

    def get_feature_list(self):
        return FEAT_DICT.values()
