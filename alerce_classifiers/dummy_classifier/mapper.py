from alerce_classifiers.base.dto import InputDTO, OutputDTO
from alerce_classifiers.base.mapper import Mapper

class DummyMapper(Mapper):
    def preprocess(self, input: InputDTO, **kwargs) -> tuple:
        return (input.detections, )
    
    def postprocess(self, model_output, **kwargs) -> OutputDTO:
        return OutputDTO(model_output)