from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic
from lc_classification.predictors.predictor.predictor_parser import PredictorOutput

T = TypeVar("T")


@dataclass
class KafkaOutput(Generic[T]):
    value: T


class KafkaParser(ABC):
    def __init__(self, class_mapper):
        self.ClassMapper = class_mapper

    @abstractmethod
    def parse(self, output: PredictorOutput, **kwargs) -> KafkaOutput:
        pass
