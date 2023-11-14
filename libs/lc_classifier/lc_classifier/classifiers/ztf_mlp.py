from typing import List

import pandas as pd
import tensorflow as tf
import os

from lc_classifier.base import AstroObject
from .base import all_features_from_astro_objects
from .base import Classifier, NotTrainedException
from .preprocess import MLPFeaturePreprocessor


class ZTFClassifier(Classifier):
    version = '1.0.0'

    def __init__(self):
        self.model = MLPModel()
        self.preprocessor = MLPFeaturePreprocessor()
        self.feature_list = None

    def classify_single_object(self, astro_object: AstroObject) -> None:
        self.classify_batch([astro_object])

    def classify_batch(self, astro_objects: List[AstroObject]) -> None:
        features_df = all_features_from_astro_objects(astro_objects)
        if self.feature_list is None:
            raise NotTrainedException('This classifier is not trained or has not been loaded')
        features_df = features_df[self.feature_list]

    def fit(self, astro_objects: List[AstroObject], labels: pd.DataFrame):
        assert len(astro_objects) == len(labels)
        all_features_df = all_features_from_astro_objects(astro_objects)
        self.fit_from_features(all_features_df, labels)

    def fit_from_features(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.feature_list = features.columns.values
        training_labels = labels[labels['partition'] == 'training']
        training_features = features.loc[training_labels['aid'].values]
        self.preprocessor.fit(training_features)
        training_features = self.preprocessor.preprocess_features(training_features)

    def save_classifier(self, directory: str):
        if not os.path.exists(directory):
            os.mkdir(directory)

        self.preprocessor.save(
            os.path.join(
                directory,
                'preprocessor.pkl'
            )
        )

    def load_classifier(self, directory: str):
        self.preprocessor.load(
            os.path.join(
                directory,
                'preprocessor.pkl'
            )
        )


class MLPModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # TODO: this is dummy code
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)

