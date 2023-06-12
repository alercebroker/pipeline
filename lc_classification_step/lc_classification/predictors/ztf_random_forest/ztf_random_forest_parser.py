from typing import List, Union
from alerce_classifiers.base.dto import OutputDTO
import pandas as pd
from lc_classification.core.parsers.input_dto import create_input_dto
from lc_classification.predictors.predictor.predictor_parser import (
    PredictorInput,
    PredictorOutput,
    PredictorParser,
)


class ZtfRandomForestParser(PredictorParser):
    def parse_input(self, to_parse: List[dict]) -> PredictorInput[pd.DataFrame]:
        dto = create_input_dto(to_parse)
        return PredictorInput(dto.features)

    def parse_output(self, to_parse: Union[dict, OutputDTO]) -> PredictorOutput:
        if isinstance(to_parse, OutputDTO):
            to_parse = {
                "probabilities": to_parse.probabilities,
                "hierarchical": {"top": pd.DataFrame(), "children": pd.DataFrame()},
            }

        return PredictorOutput(to_parse)
