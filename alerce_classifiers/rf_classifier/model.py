from abc import ABC
from alerce_base_model import ClassifierModel
from alerce_classifiers.utils.input_mapper.elasticc import ELAsTiCCMapper
from lc_classifier.classifier.models import ElasticcRandomForest

import pandas as pd


class RandomForestFeaturesClassifier(ClassifierModel, ABC):
    def __init__(self, path_to_model: str):
        self.taxonomy_dictionary = {
            "Transient": [
                "CART",
                "Iax",
                "91bg",
                "Ia",
                "Ib/c",
                "II",
                "SN-like/Other",
                "SLSN",
                "PISN",
                "TDE",
                "ILOT",
                "KN",
            ],
            "Stochastic": ["M-dwarf Flare", "Dwarf Novae", "AGN", "uLens"],
            "Periodic": ["Delta Scuti", "RR Lyrae", "Cepheid", "EB"],
        }
        self.non_used_features = [f"iqr_{band}" for band in "ugrizY"]
        super().__init__(path_to_model)

    def _load_model(self, path_to_model: str) -> None:
        self.model = ElasticcRandomForest(
            self.taxonomy_dictionary, self.non_used_features, n_trees=500, n_jobs=1
        )
        self.model.load_model(path_to_model)

    def preprocess(self, data_input: pd.DataFrame) -> pd.DataFrame:
        features = ELAsTiCCMapper.get_features(data_input)
        return features

    def predict_proba(self, data_input: pd.DataFrame) -> pd.DataFrame:
        features = self.preprocess(data_input)
        probs = self.model.predict_proba(features)
        return probs
