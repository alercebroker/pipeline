from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def compute(self, **kwargs):
        pass