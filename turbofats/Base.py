from abc import ABC, abstractmethod


class Base(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    def is1d(self):
        return True
