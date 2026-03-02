from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic
from alerce_classifiers.base.dto import OutputDTO

T = TypeVar("T")


@dataclass
class KafkaOutput(Generic[T]):
    value: T


class KafkaParser(ABC):
    def __init__(self, class_mapper):
        self.ClassMapper = class_mapper

    @abstractmethod
    def parse(self, output: OutputDTO, **kwargs) -> KafkaOutput:
        pass
