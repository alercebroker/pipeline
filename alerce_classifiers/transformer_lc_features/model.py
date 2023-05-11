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
from alerce_classifiers.utils.input_mapper.elasticc import ELAsTiCCMapper


class TransformerLCFeaturesClassifier(AlerceModel):
    def __init__(
        self, model_path: str, header_quantiles_path: str, feature_quantiles_path: str
    ):
        super().__init__(model_path)
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

    def preprocess(self, data_input: pd.DataFrame) -> pd.DataFrame:
        return self._header_classifier.preprocess(data_input)

    def preprocess_features(self, features: pd.DataFrame) -> np.ndarray:
        all_feat = []
        for col in FEATURES_ORDER:
            all_feat += [
                self.feature_quantiles[col].transform(
                    features[col].to_numpy().reshape(-1, 1)
                )
            ]
        response = np.concatenate(all_feat, 1)
        batch, num_features = response.shape
        response = response.reshape([batch, num_features, 1])
        return response

    def predict(self, data_input: pd.DataFrame) -> pd.DataFrame:
        light_curve = ELAsTiCCMapper.get_detections(data_input)
        headers = ELAsTiCCMapper.get_header(data_input, keep="first")
        headers.replace({np.nan: -9999}, inplace=True)
        features = ELAsTiCCMapper.get_features(data_input)
        features.replace({np.nan: -9999, np.inf: -9999, -np.inf: -9999}, inplace=True)
        preprocessed_light_curve = self.preprocess(light_curve)
        preprocessed_headers = self._header_classifier.preprocess_headers(headers)
        preprocessed_features = self.preprocess_features(features)

        del light_curve
        del headers

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
