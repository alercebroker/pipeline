from typing import List
import pandas as pd
from lc_classification.core.parsers.input_dto import create_input_dto
from lc_classification.predictors.predictor.predictor_parser import (
    PredictorInput,
    PredictorOutput,
    PredictorParser,
)


class ZtfRandomForestParser(PredictorParser):
    def parse_input(
        self, to_parse: List[dict], **kwargs
    ) -> PredictorInput[pd.DataFrame]:
        dto = create_input_dto(to_parse, feature_list=kwargs["feature_list"])
        return PredictorInput(dto.features)

    def parse_output(self, to_parse: dict) -> PredictorOutput:
        print(to_parse)
        return PredictorOutput(to_parse)
