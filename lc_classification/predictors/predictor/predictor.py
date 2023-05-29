import abc


class Predictor(abc.ABC):
    def __init__(self, model=None, **kwargs):
        self.model = model

    @abc.abstractmethod
    def predict(self, model_input):
        raise NotImplementedError()

    def get_feature_list(self):
        return []
