from alerce_classifiers.base.dto import InputDTO
from alerce_classifiers.base.dto import OutputDTO
from alerce_classifiers.base.model import AlerceModel
from alerce_classifiers.rf_features_header_classifier.mapper import BarneyMapper
from lc_classifier.classifier.models import ElasticcRandomForest

import pandas as pd
import validators


class RandomForestFeaturesHeaderClassifier(AlerceModel[pd.DataFrame]):
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
                "uLens",
            ],
            "Stochastic": [
                "M-dwarf Flare",
                "Dwarf Novae",
                "AGN",
            ],
            "Periodic": ["Delta Scuti", "RR Lyrae", "Cepheid", "EB"],
        }
        self.non_used_features = [f"iqr_{band}" for band in "ugrizY"]
        super().__init__(path_to_model, BarneyMapper())

    def _load_model(self, path_to_model: str) -> None:
        self.model = ElasticcRandomForest(
            self.taxonomy_dictionary,
            self.non_used_features,
            n_trees=500,
            n_jobs=1,
            model_name="barney_elasticc_rf",
        )
        if validators.url(path_to_model):
            self.model.url_model = path_to_model
            self.model.download_model()
            path_to_model = self.model.MODEL_PICKLE_PATH
        self.model.load_model(path_to_model, n_jobs=1)

    def predict(self, input_dto: InputDTO) -> OutputDTO:
        classifier_input = self.mapper.preprocess(input_dto)
        probs = self.model.predict_proba(classifier_input)
        return self.mapper.postprocess(probs)
