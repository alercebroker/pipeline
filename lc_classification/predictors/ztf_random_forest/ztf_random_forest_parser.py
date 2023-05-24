from typing import List
import pandas as pd
import numpy as np
from lc_classification.predictors.predictor.predictor_parser import (
    PredictorInput,
    PredictorOutput,
    PredictorParser,
)


class ZtfRandomForestPredictorParser(PredictorParser):
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def parse_input(self, to_parse: List[dict]) -> PredictorInput[pd.DataFrame]:
        features = self._get_features(to_parse)
        self._validate_features(features)
        parsed = PredictorInput(features)
        return parsed

    def _get_features(self, messages: List[dict]) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                {"aid": message.get("aid"), "candid": message.get("candid", np.nan)}
                for message in messages
            ]
        )
        features = pd.DataFrame([message["features"] for message in messages])
        features["aid"] = df.aid
        features["candid"] = df.candid
        features.sort_values("candid", ascending=False, inplace=True)
        features.drop_duplicates("aid", inplace=True)
        features = features.set_index("aid")
        if features is not None:
            return features
        else:
            raise ValueError("Could not set index aid on features dataframe")

    def _validate_features(self, features: pd.DataFrame):
        required_features = set(self.feature_list)
        missing_features = required_features.difference(set(features.columns))
        if len(missing_features) > 0:
            raise KeyError(
                f"Corrupted Batch: missing some features ({missing_features})"
            )

    def parse_output(self, to_parse: dict) -> PredictorOutput:
        return PredictorOutput(to_parse)
