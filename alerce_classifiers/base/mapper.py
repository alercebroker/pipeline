from abc import ABC, abstractmethod
from .dto import InputDTO

class Mapper(ABC):
    @abstractmethod
    def preprocess(input: InputDTO):
        pass

    @abstractmethod
    def postprocess():
        pass