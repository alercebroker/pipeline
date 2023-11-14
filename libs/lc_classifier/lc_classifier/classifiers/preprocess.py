from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pickle
import pandas as pd
from .base import NotTrainedException


class MLPFeaturePreprocessor:
    def __init__(self):
        self.transformer = QuantileTransformer(n_quantiles=1000, random_state=0)
        self.feature_list = None

    def inf_to_nan(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.replace([np.inf, -np.inf], np.nan)
        return features

    def fit(self, features: pd.DataFrame):
        features = self.inf_to_nan(features)
        self.transformer.fit(features.values)
        self.feature_list = features.columns.values

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(
                {
                    'feature_list': self.feature_list,
                    'transformer': self.transformer
                },
                f
            )

    def load(self, filename: str):
        with open(filename, 'rb') as f:
            attributes_dict = pickle.load(f)

        self.feature_list = attributes_dict['feature_list']
        self.transformer = attributes_dict['transformer']

    def preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_list is None:
            raise NotTrainedException(
                'fit method must be called before preprocess_features')
        features = self.inf_to_nan(features).copy()
        transformed_values = self.transformer.transform(
            features[self.feature_list].values)
        features[:] = transformed_values
        features += 0.1

        features = features.fillna(0.0)
        return features
