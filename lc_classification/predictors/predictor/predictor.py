import abc


class Predictor(abc.ABC):
    @abc.abstractmethod
    def predict(model_input):
        raise NotImplementedError()
