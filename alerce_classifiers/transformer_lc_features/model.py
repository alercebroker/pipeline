import sys
import os
import validators

import numpy as np
import pandas as pd
import torch

from joblib import load
from alerce_classifiers.base.model import AlerceModel
from alerce_classifiers.transformer_lc_features.utils import FEATURES_ORDER
from alerce_classifiers.transformer_lc_header.model import (
    TranformerLCHeaderClassifier,
)
from alerce_classifiers.utils.input_mapper.elasticc._mapper import LCFeatureMapper


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

    def to_tensor_dict(
        self, pd_output: pd.DataFrame, np_headers: np.ndarray, np_features: np.ndarray
    ):
        these_kwargs = self._header_classifier.to_tensor_dict(pd_output, np_headers)
        torch_features = torch.from_numpy(np_features).float()
        these_kwargs["tabular_feat"] = torch.cat(
            [these_kwargs["tabular_feat"], torch_features], dim=1
        )
        return these_kwargs

    def predict(self, data_input: pd.DataFrame) -> pd.DataFrame:
        (
            preprocessed_light_curve,
            preprocessed_headers,
            preprocessed_features,
        ) = self.mapper.preprocess(
            data_input,
            header_quantiles=self._header_classifier.quantiles,
            feature_quantiles=self.feature_quantiles,
        )

        input_nn = self.to_tensor_dict(
            preprocessed_light_curve, preprocessed_headers, preprocessed_features
        )
        del preprocessed_headers
        del preprocessed_features

        with torch.no_grad():
            pred = self.model.predict_mix(**input_nn)
            pred = pred["MLPMix"].exp().detach().numpy()
            preds = pd.DataFrame(
                pred, columns=self._taxonomy, index=preprocessed_light_curve.index
            )
        del input_nn
        del preprocessed_light_curve
        return preds
