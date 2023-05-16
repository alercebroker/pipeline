from abc import ABC, abstractmethod
from .dto import InputDTO

class Mapper(ABC):
    @abstractmethod
    def preprocess(input: InputDTO, **kwargs):
        pass

    @abstractmethod
    def postprocess():
        pass