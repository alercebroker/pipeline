from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import os
import pickle

import lightgbm as lgb

from base import AstroObject
from .base import all_features_from_astro_objects
from .base import Classifier, NotTrainedException
from .preprocess import MLPFeaturePreprocessor


class LightGBMClassifier(Classifier):
    version = '1.0.0'

    def __init__(self, list_of_classes: List[str]):
        self.list_of_classes = list_of_classes
        self.model = None
        self.feature_list = None
        self.preprocessor = MLPFeaturePreprocessor()

    def classify_batch(
            self,
            astro_objects: List[AstroObject],
            return_dataframe: bool = False) -> Optional[pd.DataFrame]:

        if self.feature_list is None or self.model is None:
            raise NotTrainedException(
                'This classifier is not trained or has not been loaded')
        features_df = all_features_from_astro_objects(astro_objects)
        features_df = features_df.rename(columns=lambda x: x.replace('_', '').replace(',', ''))
        features_df = features_df[self.feature_list]
        features_df = self.preprocessor.preprocess_features(
            features_df)
        probs_np = self.model.predict(features_df, num_iterations=self.model.best_iteration)
        for object_probs, astro_object in zip(probs_np, astro_objects):
            data = np.stack(
                [
                    self.list_of_classes,
                    object_probs.flatten()
                ],
                axis=-1
            )
            object_probs_df = pd.DataFrame(
                data=data,
                columns=[['name', 'value']]
            )
            object_probs_df['fid'] = None
            object_probs_df['sid'] = 'ztf'
            object_probs_df['version'] = self.version
            astro_object.predictions = object_probs_df

        if return_dataframe:
            dataframe = pd.DataFrame(
                data=probs_np,
                columns=self.list_of_classes,
                index=features_df.index
            )
            return dataframe

    def classify_batch_from_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_list is None or self.model is None:
            raise NotTrainedException(
                'This classifier is not trained or has not been loaded')
        features_df = features_df.rename(columns=lambda x: x.replace('_', '').replace(',', ''))
        features_df = features_df[self.feature_list]
        features_df = self.preprocessor.preprocess_features(
            features_df)
        probs_np = self.model.predict(features_df, num_iterations=self.model.best_iteration)

        dataframe = pd.DataFrame(
            data=probs_np,
            columns=self.list_of_classes,
            index=features_df.index
        )
        return dataframe

    def classify_single_object(self, astro_object: AstroObject) -> None:
        self.classify_batch([astro_object])

    def fit(
            self,
            astro_objects: List[AstroObject],
            labels: pd.DataFrame,
            config: Dict):

        assert len(astro_objects) == len(labels)
        all_features_df = all_features_from_astro_objects(astro_objects)
        self.fit_from_features(all_features_df, labels, config)

    def fit_from_features(
            self,
            features: pd.DataFrame,
            labels: pd.DataFrame,
            config: Dict):
        features = features.rename(columns=lambda x: x.replace('_', '').replace(',', ''))
        self.feature_list = features.columns.values

        labels = labels.copy()
        labels['astro_class_num'] = labels['astro_class'].map(
                dict(zip(self.list_of_classes, range(len(self.list_of_classes)))))

        training_labels = labels[labels['partition'] == 'training_0']
        training_features = features.loc[training_labels['aid'].values]
        self.preprocessor.fit(training_features)
        training_features = self.preprocessor.preprocess_features(
            training_features)

        validation_labels = labels[labels['partition'] == 'validation_0']
        validation_features = features.loc[validation_labels['aid'].values]
        validation_features = self.preprocessor.preprocess_features(
            validation_features)

        params = {
            'objective': 'multiclass',
            'num_class': len(self.list_of_classes),
            'metric': 'multi_logloss',
            'verbose': 1
        }

        train_data = lgb.Dataset(
            training_features,
            label=training_labels['astro_class_num'])
        valid_data = lgb.Dataset(
            validation_features,
            label=validation_labels['astro_class_num'],
            reference=train_data)

        num_round = 100
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=5),
            ],
            valid_sets=[valid_data])

    def save_classifier(self, directory: str):
        if self.model is None:
            raise NotTrainedException(
                'Cannot save model that has not been trained')

        if not os.path.exists(directory):
            os.mkdir(directory)

        self.preprocessor.save(
            os.path.join(
                directory,
                'preprocessor.pkl'
            )
        )

        with open(
                os.path.join(
                    directory,
                    'lightgbm_model.pkl'),
                'wb') as f:
            pickle.dump(
                {
                    'feature_list': self.feature_list,
                    'list_of_classes': self.list_of_classes,
                    'model': self.model
                },
                f
            )

    def load_classifier(self, directory: str):
        self.preprocessor.load(
            os.path.join(
                directory,
                'preprocessor.pkl'
            )
        )
        loaded_data = pd.read_pickle(os.path.join(directory, 'lightgbm_model.pkl'))
        self.feature_list = loaded_data['feature_list']
        self.list_of_classes = loaded_data['list_of_classes']
        self.model = loaded_data['model']
