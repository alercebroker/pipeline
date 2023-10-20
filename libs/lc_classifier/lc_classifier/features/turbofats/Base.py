from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(self, shared_data):
        self.shared_data = shared_data

    @abstractmethod
    def fit(self, data):
        pass
