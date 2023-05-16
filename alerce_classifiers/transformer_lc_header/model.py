import os
import pandas as pd
import sys
import torch
import validators

from joblib import load
from alerce_classifiers.base.dto import InputDTO
from alerce_classifiers.base.model import AlerceModel
from alerce_classifiers.utils.input_mapper.elasticc.dict_transform import FEAT_DICT

from .mapper import LCHeaderMapper


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

    def __init__(self, model_path: str, header_quantiles_path: str, mapper: LCHeaderMapper = None):
        super().__init__(model_path, mapper)
        self._local_files = f"/tmp/{type(self).__name__}"
        _file = os.path.dirname(__file__)
        sys.path.append(_file)
        self._load_quantiles(header_quantiles_path)

    def _load_quantiles(self, path: str):
        self.quantiles = {}
        existing_quantiles = FEAT_DICT.values()
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

    def predict(self, data_input: InputDTO) -> pd.DataFrame:
        input_nn = self.mapper.preprocess(data_input, quantiles=self.quantiles)

        with torch.no_grad():
            pred = self.model.predict_mix(**input_nn)
            pred = pred["MLPMix"].exp().detach().numpy()
            preds = pd.DataFrame(
                pred, columns=self._taxonomy, index=data_input.detections.index
            )
        del input_nn
        return preds

