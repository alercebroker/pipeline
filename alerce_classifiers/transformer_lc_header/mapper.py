import numpy as np
import pandas as pd
import torch
from typing import List

from alerce_classifiers.base.dto import InputDTO, OutputDTO
from alerce_classifiers.base.mapper import Mapper
from alerce_classifiers.utils.dataframe import DataframeUtils
from alerce_classifiers.utils.input_mapper.elasticc.dict_transform import FEAT_DICT

class LCHeaderMapper(Mapper):
    _fid_mapper = {
        0: "u",
        1: "g",
        2: "r",
        3: "i",
        4: "z",
        5: "Y",
    }
    _rename_cols = {
        "mag": "FLUXCAL",
        "e_mag": "FLUXCALERR",
        "fid": "BAND",
        "mjd": "MJD",
    }
    _feat_dict = FEAT_DICT

    def _get_detections(self, input: InputDTO):
        needed_cols = list(self._rename_cols.keys()).append("forced")
        return input.detections[needed_cols]

    def _get_headers(self, input: InputDTO):
        headers = pd.DataFrame.from_records(
            input.detections["extra_fields"].values,
            index=input.detections.index
        )
        headers = headers[headers["diaObject"].notnull()]
        headers = headers[~headers.index.duplicated(keep="first")]
        headers = pd.DataFrame.from_records(
            headers["diaObject"].values, index=headers.index
        )
        headers = headers[list(self._feat_dict.keys())]
        headers = headers.rename(columns=self._feat_dict)
        headers = headers.sort_index()
        headers.replace({np.nan: -9999}, inplace=True)
        return headers


    def _preprocess_detections(self, detections: pd.DataFrame):
        # Compute max epochs, maximum length per index and band
        max_epochs = DataframeUtils.get_max_epochs(detections)
        # Group by aid and creating lightcurve
        detections = detections.groupby(["aid"]).agg(lambda x: list(x))
        # Declare features that are time series
        list_time_feat = ["MJD", "FLUXCAL", "FLUXCALERR"]
        band_key = "BAND"
        # Transform features that are time series to matrices
        for key_used in list_time_feat:
            detections[key_used] = detections.apply(
                lambda x: DataframeUtils.separate_by_filter(
                    x[key_used], x[band_key], max_epochs
                ),
                axis=1,
            )
        # Normalizing time (subtract the first detection)
        detections["MJD"] = detections.apply(
            lambda x: DataframeUtils.normalizing_time(x["MJD"]), axis=1
        )

        detections["mask"] = detections.apply(
            lambda x: DataframeUtils.create_mask(x["FLUXCAL"]), axis=1
        )
        return detections

    def _preprocess_headers(self, headers: pd.DataFrame, quantiles: dict):
        all_feat = []
        for col in headers.columns:
            all_feat += [
                quantiles[col].transform(headers[col].to_numpy().reshape(-1, 1))
            ]

        response = np.concatenate(all_feat, 1)
        batch, num_features = response.shape
        response = response.reshape([batch, num_features, 1])
        return response
    
    def _to_tensor_dict(self, pd_output: pd.DataFrame, np_headers: np.ndarray) -> dict:
        torch_input = {
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
        return torch_input

    def preprocess(self, input: InputDTO, **kwargs) -> tuple:
        # TODO: obtain the forced photometry field name
        detections = self._get_detections(input)
        headers = self._get_headers(input)

        preprocessed_light_curve = self._preprocess_detections(detections)
        preprocessed_headers = self._preprocess_headers(headers, kwargs["quantiles"])
        return self._to_tensor_dict(preprocessed_light_curve, preprocessed_headers), detections.index
    
    def postprocess(self, model_output, **kwargs) -> OutputDTO:
        probs = model_output["MLPMix"].exp().detach().numpy()
        probs = pd.DataFrame(
            probs, columns=kwargs["taxonomy"], index=kwargs["index"]
        )
        return OutputDTO(probs)
