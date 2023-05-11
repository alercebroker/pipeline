from alerce_classifiers.base.model import AlerceModel
from alerce_classifiers.utils.input_mapper.elasticc import ELAsTiCCMapper
from alerce_classifiers.utils.dataframe import DataframeUtils
from joblib import load

import numpy as np
import os
import pandas as pd
import sys
import torch
import validators


class TranformerLCHeaderClassifier(AlerceModel):
    _taxonomy = [
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

    def __init__(self, model_path: str, header_quantiles_path: str):
        super().__init__(model_path)
        self._local_files = f"/tmp/{type(self).__name__}"
        _file = os.path.dirname(__file__)
        sys.path.append(_file)
        self._load_quantiles(header_quantiles_path)

    def _load_quantiles(self, path: str):
        self.quantiles = {}
        existing_quantiles = ELAsTiCCMapper.feat_dict.values()
        if validators.url(path):
            for quantile in existing_quantiles:
                quantile_url = os.path.join(path, f"norm_{quantile}.joblib")
                self.download(quantile_url, self._local_files)
            path = self._local_files

        self.quantiles = {
            quantile: load(f"{path}/norm_{quantile}.joblib")
            for quantile in existing_quantiles
        }

    def _load_model(self, model_path: str) -> None:
        if validators.url(model_path):
            model_path = self.download(model_path, self._local_files)
        self.model = torch.load(model_path, map_location=torch.device("cpu")).eval()

    def preprocess(self, data_input: pd.DataFrame) -> pd.DataFrame:
        # Compute max epochs, maximum length per index and band
        max_epochs = DataframeUtils.get_max_epochs(data_input)
        # Group by aid and creating lightcurve
        data_input = data_input.groupby(["aid"]).agg(lambda x: list(x))
        # Declare features that are time series
        list_time_feat = ["MJD", "FLUXCAL", "FLUXCALERR"]
        band_key = "BAND"
        # Transform features that are time series to matrices
        for key_used in list_time_feat:
            data_input[key_used] = data_input.apply(
                lambda x: DataframeUtils.separate_by_filter(
                    x[key_used], x[band_key], max_epochs
                ),
                axis=1,
            )
        # Normalizing time (subtract the first detection)
        data_input["MJD"] = data_input.apply(
            lambda x: DataframeUtils.normalizing_time(x["MJD"]), axis=1
        )

        data_input["mask"] = data_input.apply(
            lambda x: DataframeUtils.create_mask(x["FLUXCAL"]), axis=1
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

    @classmethod
    def to_tensor_dict(cls, pd_output: pd.DataFrame, np_headers: np.ndarray) -> dict:
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

    def predict(self, data_input: pd.DataFrame) -> pd.DataFrame:
        light_curve = ELAsTiCCMapper.get_detections(data_input)
        headers = ELAsTiCCMapper.get_header(data_input, keep="first")
        headers.replace({np.nan: -9999}, inplace=True)

        preprocessed_light_curve = self.preprocess(light_curve)
        preprocessed_headers = self.preprocess_headers(headers)
        del light_curve
        del headers

        input_nn = self.to_tensor_dict(preprocessed_light_curve, preprocessed_headers)
        del preprocessed_headers

        with torch.no_grad():
            pred = self.model.predict_mix(**input_nn)
            pred = pred["MLPMix"].exp().detach().numpy()
            preds = pd.DataFrame(
                pred, columns=self._taxonomy, index=preprocessed_light_curve.index
            )
        del input_nn
        del preprocessed_light_curve
        return preds

