from .dto import InputDTO
from .dto import OutputDTO
from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

T = TypeVar("T")


class Mapper(Generic[T], ABC):
    @abstractmethod
    def preprocess(self, input: InputDTO, **kwargs) -> T:
        pass

    @abstractmethod
    def postprocess(self, model_output, **kwargs) -> OutputDTO:
        pass
