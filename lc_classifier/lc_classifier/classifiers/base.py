from typing import Dict
from abc import ABC, abstractmethod
import pandas as pd


class NotTrainedException(Exception):
    pass


class Classifier(ABC):
    @abstractmethod
    def classify_batch(
            self,
            features: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit(self, features: pd.DataFrame, labels: pd.DataFrame, config: Dict):
        pass

    @abstractmethod
    def save_classifier(self, directory: str):
        pass

    @abstractmethod
    def load_classifier(self, directory: str):
        pass


