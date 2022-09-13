from alerce_classifiers.transformer_lc_features.utils import FEATURES_ORDER
from alerce_classifiers.transformer_lc_header import TransformerLCHeaderClassifier
from alerce_classifiers.utils.input_mapper.elasticc import ELAsTiCCMapper
from joblib import load

import numpy as np
import os
import pandas as pd
import sys
import torch
import validators


class TransformerLCFeaturesClassifier(TransformerLCHeaderClassifier):
    def __init__(
        self,
        path_to_model: str,
        path_to_header_quantiles: str,
        path_to_features_quantiles: str,
    ):
        self.local_files = f"/tmp/{type(self).__name__}/features"
        _file = os.path.dirname(__file__)
        _file = os.path.join(_file, "../transformer_lc_header")  # hack for get layers
        sys.path.append(_file)
        super().__init__(path_to_model, path_to_header_quantiles)
        self.taxonomy = [
            "AGN",
            "CART",
            "Cepheid",
            "Delta Scuti",
            "Dwarf Novae",
            "EB",
            "ILOT",
            "KN",
            "M-dwarf Flare",
            "PISN",
            "RR Lyrae",
            "SLSN",
            "91bg",
            "Ia",
            "Iax",
            "Ib/c",
            "II",
            "SN-like/Other",
            "TDE",
            "uLens",
        ]
        self._load_features_quantiles(path_to_features_quantiles)

    def _load_features_quantiles(self, path_to_quantiles: str) -> None:
        self.features_quantiles = {}
        if validators.url(path_to_quantiles):
            for val in FEATURES_ORDER:
                _val = (
                    val.replace("/", "&&&") if "/" in val else val
                )  # to read/download files with symbols in its name
                quantile_url = os.path.join(path_to_quantiles, f"norm_{_val}.joblib")
                self.download(quantile_url, self.local_files)
            path_to_quantiles = self.local_files

        for val in FEATURES_ORDER:
            _val = val.replace("/", "&&&") if "/" in val else val
            self.features_quantiles[val] = load(
                f"{path_to_quantiles}/norm_{_val}.joblib"
            )

    def to_tensor_dict_with_features(
        self, pd_output: pd.DataFrame, np_headers: np.ndarray, np_features: np.ndarray
    ) -> dict:
        these_kwargs = self.to_tensor_dict(pd_output, np_headers)
        # these_kwargs["add_tabular_feat"] = np_features
        torch_features = torch.from_numpy(np_features).float()
        these_kwargs["tabular_feat"] = torch.cat(
            [these_kwargs["tabular_feat"], torch_features], dim=1
        )
        return these_kwargs

    def preprocess(self, data_input: pd.DataFrame) -> pd.DataFrame:
        # Compute max epochs, maximum length per index and band
        max_epochs = self.get_max_epochs(data_input)
        # Group by aid and creating lightcurve
        data_input = data_input.groupby(["aid"]).agg(lambda x: list(x))
        # Declare features that are time series
        list_time_feat = ["MJD", "FLUXCAL", "FLUXCALERR"]
        # Declare band name
        band_key = "BAND"
        # Transform features that are time series to matrixes
        for key_used in list_time_feat:
            data_input[key_used] = data_input.apply(
                lambda x: self.separate_by_filter(x[key_used], x[band_key], max_epochs),
                axis=1,
            )
        # Normalizing time (subtract the first detection)
        data_input["MJD"] = data_input.apply(
            lambda x: self.normalizing_time(x["MJD"]), axis=1
        )
        # Create mask
        data_input["mask"] = data_input.apply(
            lambda x: self.create_mask(x["FLUXCAL"]), axis=1
        )
        return data_input

    def preprocess_features(self, features: pd.DataFrame) -> np.ndarray:
        all_feat = []
        for col in FEATURES_ORDER:
            all_feat += [
                self.features_quantiles[col].transform(
                    features[col].to_numpy().reshape(-1, 1)
                )
            ]

        response = np.concatenate(all_feat, 1)
        batch, num_features = response.shape
        response = response.reshape([batch, num_features, 1])
        return response

    def predict_proba(self, data_input: pd.DataFrame) -> pd.DataFrame:
        light_curve = ELAsTiCCMapper.get_detections(data_input)
        headers = ELAsTiCCMapper.get_header(data_input, keep="first")
        headers.replace({np.nan: -9999}, inplace=True)
        features = ELAsTiCCMapper.get_features(data_input)
        features.replace({np.nan: -9999}, inplace=True)

        preprocessed_light_curve = self.preprocess(light_curve)
        preprocessed_headers = self.preprocess_headers(headers)
        preprocessed_features = self.preprocess_features(features)

        del light_curve
        del headers

        input_nn = self.to_tensor_dict_with_features(
            preprocessed_light_curve, preprocessed_headers, preprocessed_features
        )
        del preprocessed_headers
        del preprocessed_features

        with torch.no_grad():
            pred = self.model.predict_mix(**input_nn)
            pred = pred["MLPMix"].exp().detach().numpy()
            preds = pd.DataFrame(
                pred, columns=self.taxonomy, index=preprocessed_light_curve.index
            )
        del input_nn
        del preprocessed_light_curve
        return preds
