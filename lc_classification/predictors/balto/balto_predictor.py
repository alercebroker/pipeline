from alerce_classifiers.transformer_lc_header.model import TransformerLCHeaderClassifier
from alerce_classifiers.transformer_lc_header.mapper import LCHeaderMapper
from alerce_classifiers.base.dto import InputDTO
from lc_classification.predictors.predictor.predictor_parser import PredictorInput
from lc_classification.predictors.predictor.predictor import Predictor
from alerce_classifiers.utils.input_mapper.elasticc.dict_transform import FEAT_DICT


class BaltoPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = str(kwargs["model_path"])
        quantiles_path = str(kwargs["quantiles_path"])
        if not self.model:
            self.model = TransformerLCHeaderClassifier(
                model_path, quantiles_path, mapper=LCHeaderMapper()
            )

    def _predict(self, model_input: PredictorInput[InputDTO]):
        return self.model.predict(model_input.value)

    def get_feature_list(self):
        return FEAT_DICT.values()
