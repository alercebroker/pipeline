from abc import ABC, abstractmethod

class IExtractor(ABC):
    @abstractmethod
    def compute(self):
        pass