import abc

from alerce_classifiers.base.dto import InputDTO, OutputDTO

from lc_classification.predictors.predictor.predictor_parser import (
    PredictorInput,
)
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

    def __init__(self, **kwargs):
        self.model = kwargs.get("model", None)

    @abc.abstractmethod
    def _predict(self, model_input: PredictorInput):
        """Wrapper for self.model's prediction method.

        The idea is that this wrapper only replaces the actual model prediction call.
        """
        raise NotImplementedError()

    def predict(self, model_input: PredictorInput):
        if self.can_predict(model_input):
            return self._predict(model_input)
        return OutputDTO(DataFrame())

    def can_predict(self, model_input: InputDTO):
        return model_input.features.any().any()

    def get_feature_list(self):
        return []
