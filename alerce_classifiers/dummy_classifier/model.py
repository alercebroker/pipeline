from abc import ABC
from alerce_classifiers.base.model import AlerceModel

import numpy as np
import pandas as pd

from .mapper import DummyMapper

class DummyClassifier(AlerceModel):
    def __init__(self, path_to_model: str):
        super().__init__(path_to_model)
        self.taxonomy = ["A", "B", "C"]
        self.mapper = DummyMapper()

    def _load_model(self, path_to_model: str) -> None:
        """
        Private method to load your model in memory. For example in torch models, you can use torch.load() for load
        the model.

        :param path_to_model: string of the path
        :return: None
        """
        self.model = None

    def predict(self, data_input: pd.DataFrame) -> pd.DataFrame:
        data_input = self.mapper.preprocess(data_input)
        probs = np.zeros((len(data_input), len(self.taxonomy)))
        response = pd.DataFrame(probs, columns=self.taxonomy)
        return self.mapper.postprocess(response)
