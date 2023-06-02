import abc

from alerce_classifiers.base.dto import InputDTO, OutputDTO

from lc_classification.predictors.predictor.predictor_parser import PredictorInput
from pandas import DataFrame


class Predictor(abc.ABC):
    """Wrapper for models.

    Abstract class that wraps the predict functionality of different
    ML model implementations.

    The implementations of this abstract class should implement the _predict method.

    You should interact with this class calling the predict() method.

    Optionally, you can override the can_predict method to suit your actual model needs.

    By default, predict will return an empty list if no classification was made.
    """

    def __init__(self, model=None, **kwargs):
        self.model = model

    @abc.abstractmethod
    def _predict(self, model_input: PredictorInput):
        raise NotImplementedError()

    def predict(self, model_input: PredictorInput):
        if self.can_predict(model_input):
            return self._predict(model_input)
        return OutputDTO(DataFrame())

    def can_predict(self, model_input: PredictorInput[InputDTO]):
        return model_input.value.features.any().any()

    def get_feature_list(self):
        return []
