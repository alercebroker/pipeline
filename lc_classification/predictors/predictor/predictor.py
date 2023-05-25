import abc


class Predictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, model_input):
        raise NotImplementedError()

    def get_feature_list(self):
        return []
