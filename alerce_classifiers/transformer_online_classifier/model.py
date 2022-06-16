from abc import ABC
from alerce_base_model import ClassifierModel

import numpy as np
import os
import pandas as pd
import sys
import torch


class TransformerOnlineClassifier(ClassifierModel, ABC):
    def __init__(self, path_to_model: str):
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

    def get_max_epochs(self, pd_output):
        return pd_output.groupby(["aid", "BAND"]).count()["FLUXCAL"].max()

    def pad_list(self, lc, nepochs, max_epochs):
        pad_num = max_epochs - nepochs
        if pad_num >= 0:
            return np.pad(lc, (0, pad_num), "constant", constant_values=(0, 0))
        else:
            return np.array(lc)[np.linspace(0, nepochs - 1, num=max_epochs).astype(int)]

    def create_mask(self, lc):
        return (lc != 0).astype(float)

    def normalizing_time(self, time_fid):
        mask_min = 9999999999 * (time_fid == 0).astype(float)
        t_min = np.min(time_fid + mask_min)
        return (time_fid - t_min) * (~(time_fid == 0)).astype(float)

    def separate_by_filter(self, feat_time_series, bands, max_epochs):
        timef_array = np.array(feat_time_series)
        band_array = np.array(bands)
        colors = {"u": "b", "g": "g", "r": "r", "i": "orange", "z": "brown", "Y": "k"}
        final_array = []
        for i, color in enumerate(colors.keys()):
            aux = timef_array[band_array == color]
            nepochs = len(aux)
            final_array += [self.pad_list(aux, nepochs, max_epochs)]
        return np.stack(final_array, 1)

    def to_tensor_dict(self, pd_output: pd.DataFrame) -> dict:
        these_kwargs = {
            "data": torch.from_numpy(
                np.stack(pd_output["FLUXCAL"].to_list(), 0)
            ).float(),
            "data_var": torch.from_numpy(
                np.stack(pd_output["FLUXCALERR"].to_list(), 0)
            ).float(),
            "time": torch.from_numpy(np.stack(pd_output["MJD"].to_list(), 0)).float(),
            "mask": torch.from_numpy(np.stack(pd_output["mask"].to_list(), 0)).float(),
        }
        return these_kwargs

    def _load_model(self, path_to_model: str) -> None:
        self.model = torch.load(path_to_model, map_location=torch.device("cpu")).eval()

    def preprocess(self, data_input: pd.DataFrame) -> pd.DataFrame:
        # Compute max epochs, maximum length per index and band
        max_epochs = self.get_max_epochs(data_input)
        # Groupby by aid and creating lightcurve
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

    def predict_proba(self, data_input: pd.DataFrame) -> pd.DataFrame:
        preprocessed = self.preprocess(data_input)
        input_nn = self.to_tensor_dict(preprocessed)
        pred = self.model.predict(**input_nn)
        preds = pd.DataFrame(
            pred.numpy(), columns=self.taxonomy, index=preprocessed.index
        )
        return preds
