from abc import ABC, abstractmethod
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pickle
import pandas as pd
from .base import NotTrainedException
from typing import List


def inf_to_nan(features: pd.DataFrame) -> pd.DataFrame:
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


class FeaturePreprocessor(ABC):
    @abstractmethod
    def preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        pass


class MLPFeaturePreprocessor(FeaturePreprocessor):
    def __init__(self):
        self.transformer = QuantileTransformer(n_quantiles=1000, random_state=0)
        self.feature_list = None

    def fit(self, features: pd.DataFrame):
        features = inf_to_nan(features)
        self.transformer.fit(features.values)
        self.feature_list = features.columns.values

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(
                {"feature_list": self.feature_list, "transformer": self.transformer}, f
            )

    def load(self, filename: str):
        with open(filename, "rb") as f:
            attributes_dict = pickle.load(f)

        self.feature_list = attributes_dict["feature_list"]
        self.transformer = attributes_dict["transformer"]

    def preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_list is None:
            raise NotTrainedException(
                "fit method must be called before preprocess_features"
            )
        features = inf_to_nan(features).copy()
        transformed_values = self.transformer.transform(
            features[self.feature_list].values
        )
        features[:] = transformed_values
        features += 0.1

        features = features.fillna(0.0)
        return features


class RandomForestPreprocessor(FeaturePreprocessor):
    def preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        features = inf_to_nan(features.astype(np.float64)).copy()
        features = features.fillna(-999.0)
        features.columns = self.preprocess_feature_names(features.columns.values)
        return features

    def preprocess_feature_names(self, feature_names: List[str]):
        new_features = []
        for feature in feature_names:
            feature = feature.replace("-", "_")
            if feature == "ps_g_r":
                new_features.append(feature)
                continue

            splitted_feature = feature.split("_")
            if len(splitted_feature) == 1:
                new_features.append(feature)
                continue

            feature_root = "_".join(splitted_feature[:-1])
            feature_ending = splitted_feature[-1]
            if feature_ending == "g":
                feature_root += "_1"
            elif feature_ending == "r":
                feature_root += "_2"
            elif feature_ending == "g,r":
                feature_root += "_12"
            elif feature_ending == "nan":
                pass
            else:
                feature_root += "_" + feature_ending
            new_features.append(feature_root)
        return new_features
