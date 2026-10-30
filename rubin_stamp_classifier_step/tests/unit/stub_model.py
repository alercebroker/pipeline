"""A model the step can load by import path without any artifact.

Used by the unit tests through MODEL_CONFIG.CLASS. It returns the same
probability row for every oid it is asked about and records what it was
built with and what it received.
"""
import pandas as pd

from alerce_classifiers.base.dto import OutputDTO

CLASSES = ["AGN", "SN", "VS", "asteroid", "bogus"]
MODEL_VERSION = "1.0.0"


class StubModel:
    ROW = {"AGN": 0.1, "SN": 0.6, "VS": 0.1, "asteroid": 0.1, "bogus": 0.1}

    def __init__(self, **params):
        self.params = params
        self.dict_mapping_classes = dict(enumerate(CLASSES))
        self.model_version = MODEL_VERSION
        self.calls = []

    def predict(self, input_dto) -> OutputDTO:
        index = input_dto.stamps.index
        sids = input_dto.features.loc[index, "sid"].tolist()
        self.calls.append(list(zip(index.tolist(), sids)))
        probs = pd.DataFrame([self.ROW] * len(index), index=index, columns=CLASSES)
        return OutputDTO(probabilities=probs, hierarchical=None)
