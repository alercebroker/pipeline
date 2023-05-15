from alerce_classifiers.base.model import AlerceModel
from alerce_classifiers.utils.input_mapper.elasticc._mapper import LCHeaderMapper
from alerce_classifiers.utils.input_mapper.elasticc import ELAsTiCCMapper
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

    def __init__(self, model_path: str, header_quantiles_path: str, mapper: LCHeaderMapper):
        super().__init__(model_path, mapper)
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
        light_curve, headers = self.mapper.preprocess(data_input, quantiles=self.quantiles)
        input_nn = self.to_tensor_dict(light_curve, headers)
        del headers

        with torch.no_grad():
            pred = self.model.predict_mix(**input_nn)
            pred = pred["MLPMix"].exp().detach().numpy()
            preds = pd.DataFrame(
                pred, columns=self._taxonomy, index=light_curve.index
            )
        del input_nn
        del light_curve
        return preds

