from alerce_classifiers.base.dto import InputDTO, OutputDTO
from alerce_classifiers.transformer_lc_features.mapper import LCFeatureMapper

class RandomForestClassifierMapper(LCFeatureMapper):
    def preprocess(self, input: InputDTO, **kwargs) -> tuple:
        return (self._get_features(input), )
    
    def postprocess(self, model_output, **kwargs) -> OutputDTO:
        return OutputDTO(model_output)