from abc import ABC, abstractmethod
from .dto import InputDTO, OutputDTO

class Mapper(ABC):
    @abstractmethod
    def preprocess(self, input: InputDTO, **kwargs) -> tuple:
        pass

    @abstractmethod
    def postprocess(self, model_output, **kwargs) -> OutputDTO:
        pass