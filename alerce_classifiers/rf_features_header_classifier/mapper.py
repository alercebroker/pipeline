from alerce_classifiers.base.dto import InputDTO
from alerce_classifiers.base.dto import OutputDTO
from alerce_classifiers.base.mapper import Mapper
from alerce_classifiers.utils.input_mapper.elasticc.dict_transform import FEAT_DICT

import numpy as np
import pandas as pd


class RandomForestClassifierMapper(Mapper):
    feat_dict = FEAT_DICT

    def _get_headers_from_detections(self, detections: pd.DataFrame) -> pd.DataFrame:
        detections = detections.sort_values(by=["mjd"])
        headers = pd.DataFrame.from_records(
            detections["extra_fields"].values, index=detections.index
        )
        headers = headers[headers["diaObject"].notnull()]
        headers = headers[~headers.index.duplicated(keep="first")]
        headers = pd.DataFrame.from_records(
            headers["diaObject"].values, index=headers.index
        )
        headers = headers[list(self.feat_dict.keys())]
        headers = headers.rename(columns=self.feat_dict)
        return headers.sort_index()

    def preprocess(self, input: InputDTO, **kwargs) -> tuple:
        features = input.features.replace({None: np.nan})
        headers = self._get_headers_from_detections(input.detections)
        return (headers, features)

    def postprocess(self, model_output, **kwargs) -> OutputDTO:
        return OutputDTO(model_output)
