import abc
from datetime import datetime
from typing import List

import pandas as pd


class BaseStrategy(abc.ABC):
    def __init__(self, name, version):
        self.name = name
        self.version = version

    @staticmethod
    def _sort_probabilities(obj_probabilities: dict) -> List[tuple]:
        return sorted(obj_probabilities.items(), key=lambda x: x[1], reverse=True)

    def _with_ranking(self, raw_probabilities: dict):
        return {
            aid: {
                "classifier_name": self.name,
                "classifier_version": self.version,
                "class_name": cls,
                "probability": prob,
                "ranking": i + 1,
            }
            for aid, obj in raw_probabilities.items()
            for i, (cls, prob) in enumerate(self._sort_probabilities(obj))
        }

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def _to_dataframe(self, messages: List[dict]) -> pd.DataFrame:
        pass

    def get_probabilities(self, messages: List[dict]) -> dict:
        return self._with_ranking(
            self.predict(self._to_dataframe(messages)).to_dict(orient="index")
        )
