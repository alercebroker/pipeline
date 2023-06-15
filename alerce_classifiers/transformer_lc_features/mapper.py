import numpy as np
import pandas as pd
import torch
from alerce_classifiers.base.dto import InputDTO
from alerce_classifiers.balto.mapper import BaltoMapper
from alerce_classifiers.transformer_lc_features.utils import FEATURES_ORDER


class LCFeatureMapper(BaltoMapper):
    def _get_features(self, input: InputDTO):
        features = input.features
        return features.replace({None: np.nan})

    def _preprocess_features(self, features: pd.DataFrame, feature_quantiles: dict):
        features.replace({np.nan: -9999, np.inf: -9999, -np.inf: -9999}, inplace=True)
        all_feat = []
        for col in FEATURES_ORDER:
            all_feat += [
                feature_quantiles[col].transform(
                    features[col].to_numpy().reshape(-1, 1)
                )
            ]
        response = np.concatenate(all_feat, 1)
        batch, num_features = response.shape
        response = response.reshape([batch, num_features, 1])
        return response

    def _feat_to_tensor_dict(self, torch_input, np_features: np.ndarray) -> dict:
        torch_features = torch.from_numpy(np_features).float()
        torch_input["tabular_feat"] = torch.cat(
            [torch_input["tabular_feat"], torch_features], dim=1
        )
        return torch_input

    def preprocess(self, input: InputDTO, **kwargs) -> tuple:
        features = self._get_features(input)
        preprocessed_features = self._preprocess_features(
            features, kwargs["feature_quantiles"]
        )
        torch_input, aid_index = super().preprocess(
            input, quantiles=kwargs["header_quantiles"]
        )
        return self._feat_to_tensor_dict(torch_input, preprocessed_features), aid_index
