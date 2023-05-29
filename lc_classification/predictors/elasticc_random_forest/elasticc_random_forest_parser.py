from alerce_classifiers.base.dto import InputDTO, OutputDTO
from pandas import DataFrame
from lc_classification.core.parsers.input_dto import create_input_dto
from lc_classification.predictors.predictor.predictor_parser import (
    PredictorInput,
    PredictorOutput,
    PredictorParser,
)


class ElasticcRandomForestParser(PredictorParser):
    def parse_input(self, to_parse, **kwargs) -> PredictorInput[InputDTO]:
        dto = create_input_dto(to_parse, **kwargs)
        # features_as_list = dto.features.to_dict(orient="records")
        # features_for_model = DataFrame(
        #     {"features": features_as_list}, index=dto.features.index
        # )
        return PredictorInput(dto)

    def parse_output(self, to_parse: OutputDTO, **kwargs) -> PredictorOutput:
        result = {
            "probabilities": to_parse.probabilities,
            "hierarchical": {"top": DataFrame(), "children": DataFrame()},
        }
        return PredictorOutput(result)
