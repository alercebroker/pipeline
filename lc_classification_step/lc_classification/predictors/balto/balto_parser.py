from alerce_classifiers.base.dto import InputDTO, OutputDTO
from pandas import DataFrame
from lc_classification.core.parsers.input_dto import create_input_dto
from lc_classification.predictors.predictor.predictor_parser import (
    PredictorInput,
    PredictorOutput,
    PredictorParser,
)


class BaltoParser(PredictorParser):
    def parse_input(self, to_parse, **kwargs) -> InputDTO:
        dto = create_input_dto(to_parse, **kwargs)
        return dto

    def parse_output(self, to_parse: OutputDTO) -> PredictorOutput:
        result = {
            "probabilities": to_parse.probabilities,
            "hierarchical": {"top": DataFrame(), "children": DataFrame()},
        }
        return PredictorOutput(result)
