from abc import ABC
from alerce_base_model import ClassifierModel
from alerce_classifiers.utils.input_mapper.elasticc import ELAsTiCCMapper
from joblib import load

import numpy as np
import os
import pandas as pd
import sys
import torch
import validators


class TransformerLCHeaderClassifier(ClassifierModel, ABC):
    def __init__(self, path_to_model: str, path_to_quantiles: str):
        self.local_files = f"/tmp/{type(self).__name__}"
        _file = os.path.dirname(__file__)
        sys.path.append(_file)
        super().__init__(path_to_model)
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
        self._load_quantiles(path_to_quantiles)

    def _load_model(self, path_to_model: str) -> None:
        if validators.url(path_to_model):
            path_to_model = self.download(path_to_model, self.local_files)
        self.model = torch.load(path_to_model, map_location=torch.device("cpu")).eval()

    def _load_quantiles(self, path_to_quantiles: str) -> None:
        self.quantiles = {}
        existing_quantiles = ELAsTiCCMapper.feat_dict.items()

        if validators.url(path_to_quantiles):
            for key, val in existing_quantiles:
                quantile_url = os.path.join(path_to_quantiles, f"norm_{val}.joblib")
                self.download(quantile_url, self.local_files)
            path_to_quantiles = self.local_files

        for key, val in ELAsTiCCMapper.feat_dict.items():
            self.quantiles[val] = load(f"{path_to_quantiles}/norm_{val}.joblib")

    def get_max_epochs(self, pd_output: pd.DataFrame) -> int:
        return pd_output.groupby(["aid", "BAND"]).count()["FLUXCAL"].max()

    def pad_list(self, lc: pd.DataFrame, nepochs: int, max_epochs: int) -> np.ndarray:
        pad_num = max_epochs - nepochs
        if pad_num >= 0:
            return np.pad(lc, (0, pad_num), "constant", constant_values=(0, 0))
        else:
            return np.array(lc)[np.linspace(0, nepochs - 1, num=max_epochs).astype(int)]

    def create_mask(self, lc: pd.DataFrame) -> np.ndarray:
        return (lc != 0).astype(float)

    def normalizing_time(self, time_fid: pd.Series) -> pd.Series:
        mask_min = 9999999999 * (time_fid == 0).astype(float)
        t_min = np.min(time_fid + mask_min)
        return (time_fid - t_min) * (~(time_fid == 0)).astype(float)

    def separate_by_filter(
        self, feat_time_series: np.ndarray, bands: np.ndarray, max_epochs: int
    ) -> np.ndarray:
        timef_array = np.array(feat_time_series)
        band_array = np.array(bands)
        colors = {"u": "b", "g": "g", "r": "r", "i": "orange", "z": "brown", "Y": "k"}
        final_array = []
        for i, color in enumerate(colors.keys()):
            aux = timef_array[band_array == color]
            nepochs = len(aux)
            final_array += [self.pad_list(aux, nepochs, max_epochs)]
        return np.stack(final_array, 1)

    def to_tensor_dict(self, pd_output: pd.DataFrame, np_headers: np.ndarray) -> dict:
        these_kwargs = {
            "data": torch.from_numpy(
                np.stack(pd_output["FLUXCAL"].to_list(), 0)
            ).float(),
            "data_var": torch.from_numpy(
                np.stack(pd_output["FLUXCALERR"].to_list(), 0)
            ).float(),
            "time": torch.from_numpy(np.stack(pd_output["MJD"].to_list(), 0)).float(),
            "mask": torch.from_numpy(np.stack(pd_output["mask"].to_list(), 0)).float(),
            "tabular_feat": torch.from_numpy(np_headers).float(),
        }
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

    def preprocess_headers(self, headers: pd.DataFrame) -> np.ndarray:
        all_feat = []
        for col in headers.columns:
            all_feat += [
                self.quantiles[col].transform(headers[col].to_numpy().reshape(-1, 1))
            ]

        response = np.concatenate(all_feat, 1)
        batch, num_features = response.shape
        response = response.reshape([batch, num_features, 1])
        return response

    def predict_proba(self, data_input: pd.DataFrame) -> pd.DataFrame:
        detections = ELAsTiCCMapper.get_detections(data_input)
        headers = ELAsTiCCMapper.get_header(data_input, keep="first")
        headers.replace({np.nan: -9999}, inplace=True)

        preprocessed_detections = self.preprocess(detections)
        preprocessed_headers = self.preprocess_headers(headers)

        input_nn = self.to_tensor_dict(preprocessed_detections, preprocessed_headers)
        pred = self.model.predict_mix(**input_nn)
        preds = pd.DataFrame(
            pred.numpy(), columns=self.taxonomy, index=preprocessed_detections.index
        )
        return preds
