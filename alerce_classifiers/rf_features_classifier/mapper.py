from alerce_classifiers.base.dto import InputDTO, OutputDTO
from alerce_classifiers.base.mapper import Mapper
import numpy as np


class RandomForestClassifierMapper(Mapper):
    def preprocess(self, input: InputDTO, **kwargs) -> tuple:
        features = input.features
        return (features.replace({None: np.nan}),)

    def postprocess(self, model_output, **kwargs) -> OutputDTO:
        return OutputDTO(model_output)
