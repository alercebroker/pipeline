from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic
from lc_classification.predictors.predictor.predictor_parser import PredictorOutput

T = TypeVar("T")


@dataclass
class KafkaOutput(Generic[T]):
    value: T


class KafkaParser(ABC):
    @abstractmethod
    def parse(self, output: PredictorOutput, **kwargs) -> KafkaOutput:
        pass
