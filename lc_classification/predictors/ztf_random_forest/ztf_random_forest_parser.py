from typing import List
import pandas as pd
import numpy as np
from lc_classification.core.parsers.input_dto import create_input_dto
from lc_classification.predictors.predictor.predictor_parser import (
    PredictorInput,
    PredictorOutput,
    PredictorParser,
)


class ZtfRandomForestPredictorParser(PredictorParser):
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def parse_input(self, to_parse: List[dict]) -> PredictorInput[pd.DataFrame]:
        dto = create_input_dto(to_parse, feature_list=self.feature_list)
        return PredictorInput(dto.features)

    def _validate_features(self, features: pd.DataFrame):
        required_features = set(self.feature_list)
        missing_features = required_features.difference(set(features.columns))
        if len(missing_features) > 0:
            raise KeyError(
                f"Corrupted Batch: missing some features ({missing_features})"
            )

    def parse_output(self, to_parse: dict) -> PredictorOutput:
        return PredictorOutput(to_parse)
