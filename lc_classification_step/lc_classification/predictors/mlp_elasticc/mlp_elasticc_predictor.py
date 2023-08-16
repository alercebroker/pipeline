from alerce_classifiers.base.dto import InputDTO
from alerce_classifiers.mlp_elasticc.model import MLPElasticcClassifier
from ..predictor.predictor import Predictor


class MLPElasticcPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = str(kwargs["model_path"])
        if not self.model:
            self.model = MLPElasticcClassifier(model_path)

    def _predict(self, model_input: InputDTO):
        return self.model.predict(model_input)

    def get_feature_list(self):
        return self.model.model.feature_preprocessor.feature_list
