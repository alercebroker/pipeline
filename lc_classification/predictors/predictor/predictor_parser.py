import abc
from dataclasses import dataclass
from typing import Generic, List, TypeVar

T = TypeVar("T")


@dataclass
class PredictorInput(Generic[T]):
    value: T


@dataclass
class Classification:
    aid: str
    classification: dict


@dataclass
class PredictorOutput:
    classifications: List[Classification]


class PredictorParser(abc.ABC):
    @abc.abstractmethod
    def parse_input(to_parse) -> PredictorInput:
        raise NotImplementedError()

    @abc.abstractmethod
    def parse_output(to_parse) -> PredictorOutput:
        raise NotImplementedError()
