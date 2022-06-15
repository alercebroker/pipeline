import numpy as np
import pandas as pd

from abc import ABC
from alerce_base_model import ClassifierModel


class DummyClassifier(ClassifierModel, ABC):
    def __init__(self, path_to_model: str):
        super().__init__(path_to_model)
        self.taxonomy = ["A", "B", "C"]

    def _load_model(self, path_to_model: str) -> None:
        """
        Private method to load your model in memory. For example in torch models, you can use torch.load() for load
        the model.

        :param path_to_model: string of the path
        :return: None
        """
        self.model = None

    def preprocess(self, data_input: pd.DataFrame) -> pd.DataFrame:
        """
        Method for preprocess input data. You need to implement transformation, filters or custom processes to
        preprocess the data.

        :param data_input: DataFrame with input data
        :return: DataFrame of preprocessed input data
        """
        return data_input

    def predict_proba(self, data_input: pd.DataFrame) -> pd.DataFrame:
        data_input = self.preprocess(data_input)
        probs = np.zeros((len(data_input), len(self.taxonomy)))
        response = pd.DataFrame(probs, columns=self.taxonomy)
        return response

