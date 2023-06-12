import abc
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class PredictorInput(Generic[T]):
    value: T


@dataclass
class PredictorOutput:
    classifications: dict

    def __post_init__(self):
        assert "probabilities" in self.classifications


class PredictorParser(abc.ABC):
    @abc.abstractmethod
    def parse_input(self, to_parse, **kwargs) -> PredictorInput:
        raise NotImplementedError()

    @abc.abstractmethod
    def parse_output(self, to_parse, **kwargs) -> PredictorOutput:
        raise NotImplementedError()
