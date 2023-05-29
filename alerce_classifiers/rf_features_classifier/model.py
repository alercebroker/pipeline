from alerce_classifiers.base.model import AlerceModel
from alerce_classifiers.base.dto import InputDTO, OutputDTO
from lc_classifier.classifier.models import ElasticcRandomForest

import validators

from .mapper import RandomForestClassifierMapper


class RandomForestFeaturesClassifier(AlerceModel):
    def __init__(self, path_to_model: str):
        self._taxonomy_dictionary = {
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
        self._non_used_features = [f"iqr_{band}" for band in "ugrizY"]
        self.feature_list = None
        super().__init__(path_to_model, RandomForestClassifierMapper())

    def _load_model(self, path_to_model: str) -> None:
        self.model = ElasticcRandomForest(
            self._taxonomy_dictionary, self._non_used_features, n_trees=500, n_jobs=1
        )
        if validators.url(path_to_model):
            self.model.url_model = path_to_model
            self.model.download_model()
            path_to_model = self.model.MODEL_PICKLE_PATH
        self.model.load_model(path_to_model, n_jobs=1)
        self.feature_list = []
        for feat_group in self.model.feature_list_dict.values():
            self.feature_list.extend(feat_group)
            self.feature_list = list(set(self.feature_list))

    def predict(self, input_dto: InputDTO) -> OutputDTO:
        features = self.mapper.preprocess(input_dto)[0]
        probs = self.model.predict_proba(features)
        return self.mapper.postprocess(probs)
