from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import os
import pickle
from imblearn.ensemble import BalancedRandomForestClassifier

from lc_classifier.base import AstroObject
from .base import all_features_from_astro_objects
from .base import Classifier, NotTrainedException
from .preprocess import RandomForestPreprocessor


class HierarchicalRandomForestClassifier(Classifier):
    version = '2.0.0'

    def __init__(self, list_of_classes: List[str]):
        self.list_of_classes = list_of_classes
        self.model = None
        self.feature_list = None
        self.preprocessor = RandomForestPreprocessor()

    def classify_batch(
            self,
            astro_objects: List[AstroObject],
            return_dataframe: bool = False) -> Optional[pd.DataFrame]:

        if self.feature_list is None or self.model is None:
            raise NotTrainedException(
                'This classifier is not trained or has not been loaded')
        features_df = all_features_from_astro_objects(astro_objects)
        features_df = features_df[self.feature_list]
        features_np = self.preprocessor.preprocess_features(
            features_df).values
        probs_np = self.model.predict_proba(features_np)
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
        features_df = features_df[self.feature_list]
        features_np = self.preprocessor.preprocess_features(
            features_df).values
        probs_np = self.model.predict_proba(features_np)

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
        self.feature_list = features.columns.values
        training_labels = labels[labels['partition'] == 'training_0']
        training_features = features.loc[training_labels['aid'].values]
        training_features = self.preprocessor.preprocess_features(
            training_features)

        top_model = BalancedRandomForestClassifier(
            n_estimators=config['n_trees'],
            max_depth=None,
            max_features="sqrt",
            n_jobs=config['n_jobs'],
            verbose=config['verbose']
        )

        stochastic_model = BalancedRandomForestClassifier(
            n_estimators=config['n_trees'],
            max_depth=None,
            max_features=0.2,
            n_jobs=config['n_jobs'],
            verbose=config['verbose']
        )

        periodic_classifier = BalancedRandomForestClassifier(
            n_estimators=config['n_trees'],
            max_depth=None,
            max_features="sqrt",
            n_jobs=config['n_jobs'],
            verbose=config['verbose']
        )

        transient_model = BalancedRandomForestClassifier(
            n_estimators=config['n_trees'],
            max_depth=None,
            max_features="sqrt",
            n_jobs=config['n_jobs'],
            verbose=config['verbose']
        )

        self.model = {
            'top': top_model,
            'stochastic': stochastic_model,
            'periodic': periodic_classifier,
            'transient': transient_model
        }
        # TODO work in progress

        self.model.fit(
            training_features.values,
            training_labels['astro_class'].values)

    def save_classifier(self, directory: str):
        if self.model is None:
            raise NotTrainedException(
                'Cannot save model that has not been trained')

        if not os.path.exists(directory):
            os.mkdir(directory)

        with open(
                os.path.join(
                    directory,
                    'random_forest_model.pkl'),
                'wb') as f:
            pickle.dump(
                {
                    'feature_list': self.feature_list,
                    'list_of_classes': self.list_of_classes,
                    'random_forest': self.model
                },
                f
            )

    def load_classifier(self, directory: str):
        loaded_data = pd.read_pickle(os.path.join(directory, 'random_forest_model.pkl'))
        self.feature_list = loaded_data['feature_list']
        self.list_of_classes = loaded_data['list_of_classes']
        self.model = loaded_data['random_forest']
