import sys
import os
import validators

import numpy as np
import pandas as pd
import torch

from joblib import load
from alerce_classifiers.base.dto import InputDTO, OutputDTO
from alerce_classifiers.base.model import AlerceModel
from alerce_classifiers.transformer_lc_features.utils import FEATURES_ORDER
from alerce_classifiers.transformer_lc_header.model import (
    TranformerLCHeaderClassifier,
)
from .mapper import LCFeatureMapper


class TransformerLCFeaturesClassifier(AlerceModel):
    def __init__(
        self, model_path: str, header_quantiles_path: str, feature_quantiles_path: str, mapper: LCFeatureMapper
    ):
        super().__init__(model_path, mapper)
        self.local_files = f"/tmp/{type(self).__name__}/features"
        # some ugly hack
        sys.path.append(
            os.path.join(os.path.dirname(__file__), "../transformer_lc_header")
        )
        self._header_classifier = TranformerLCHeaderClassifier(
            model_path, header_quantiles_path
        )
        self._taxonomy = self._header_classifier._taxonomy
        self._load_feature_quantiles(feature_quantiles_path)

    def _load_feature_quantiles(self, path):
        self.feature_quantiles = {}
        if validators.url(path):
            for feat in FEATURES_ORDER:
                parsed_feat = feat.replace("/", "&&&")
                url = os.path.join(path, f"norm_{parsed_feat}.joblib")
                self.download(url, self.local_files)
            path = self.local_files

        for feat in FEATURES_ORDER:
            parsed_feat = feat.replace("/", "&&&")
            self.feature_quantiles[feat] = load(f"{path}/norm_{parsed_feat}.joblib")

    def predict(self, data_input: InputDTO) -> OutputDTO:
        input_nn, aid_index = self.mapper.preprocess(
            data_input,
            header_quantiles=self._header_classifier.quantiles,
            feature_quantiles=self.feature_quantiles,
        )

        with torch.no_grad():
            pred = self.model.predict_mix(**input_nn)

        return self.mapper.postprocess(pred, index=aid_index)
