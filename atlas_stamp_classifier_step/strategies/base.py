import abc
import os
from typing import List

import pandas as pd


class BaseStrategy(abc.ABC):
    """Strategy base class.

    Derived classes should instantiate their own classifier and implement the abstract methods:

    * `to_dataframe`: Turns the messages into a pandas `DataFrame` with the columns expected by the model
    * `predict`: Call the prediction method of the model using the data frame mentioned above

    The last method should be simple, but is required since the prediction method names are not standardized

    NOTE:
        The `__init__` method here only receives defaults for the name and version of the classifier. The
        `__init__` method of subclasses is recommended to be left without arguments, but values can also be passed
        in the function `get_strategy` (found in the `__init__.py` file in this folder)
    """

    def __init__(self, default_name, default_version):
        self.name = os.getenv("MODEL_NAME", default_name)
        self.version = os.getenv("MODEL_VERSION", default_version)

    @staticmethod
    def _sort_probabilities(obj_probabilities: dict) -> List[tuple]:
        return sorted(obj_probabilities.items(), key=lambda x: x[1], reverse=True)

    def _with_ranking(self, raw_probabilities: dict):
        return {
            aid: [
                {
                    "classifier_name": self.name,
                    "classifier_version": self.version,
                    "class_name": cls,
                    "probability": prob,
                    "ranking": i + 1,
                }
                for i, (cls, prob) in enumerate(self._sort_probabilities(obj))
            ]
            for aid, obj in raw_probabilities.items()
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
