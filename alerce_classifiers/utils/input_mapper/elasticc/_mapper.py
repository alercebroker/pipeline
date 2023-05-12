import numpy as np
import pandas as pd
from alerce_classifiers.base.dto import InputDTO

from alerce_classifiers.base.mapper import Mapper
from .dict_transform import FEAT_DICT

class ElasticcMapper(Mapper):
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
        return headers

    def preprocess(self, input: InputDTO):
        # TODO: obtain the forced photometry field name
        detections = self._get_detections(input)
        headers = self._get_headers(input)
        features = input.features
        features.replace({ None: np.nan }, inplace=True)

        return detections, headers, features
    
    def postprocess(self):
        pass