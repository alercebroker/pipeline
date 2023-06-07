from alerce_classifiers.base.dto import InputDTO
from alerce_classifiers.base.dto import OutputDTO
from alerce_classifiers.base.mapper import Mapper

import numpy as np
import pandas as pd


class TorettoMapper(Mapper[pd.DataFrame]):
    def preprocess(self, input: InputDTO, **kwargs) -> pd.DataFrame:
        features = input.features
        return features.replace({None: np.nan})

    def postprocess(self, model_output, **kwargs) -> OutputDTO:
        return OutputDTO(model_output)
