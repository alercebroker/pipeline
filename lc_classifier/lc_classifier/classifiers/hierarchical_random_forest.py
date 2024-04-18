from typing import List, Dict, Union, Tuple

import pandas as pd
import os
import pickle
from imblearn.ensemble import BalancedRandomForestClassifier

from .base import Classifier, NotTrainedException
from .preprocess import RandomForestPreprocessor


class HierarchicalRandomForestClassifier(Classifier):
    version = '2.0.0'
    class_hierarchy = {
        'transient': [
            'SNIa',
            'SNIbc',
            'SNIIb',
            'SNII',
            'SNIIn',
            'SLSN',
            'TDE'
        ],
        'periodic': [
            'LPV',
            'EA',
            'EB/EW',
            'Periodic-Other',
            'RSCVn',
            'CEP',
            'RRLab',
            'RRLc',
            'DSCT'
        ],
        'stochastic': [
            'QSO',
            'AGN',
            'Blazar',
            'YSO',
            'CV/Nova',
            'Microlensing'  # ulens get confused with stochastic classes
        ]
    }

    def __init__(self, list_of_classes: List[str]):
        self.list_of_classes = list_of_classes
        self.model = None
        self.feature_list = None
        self.preprocessor = RandomForestPreprocessor()

    def classify_batch(
            self,
            features: pd.DataFrame,
            return_hierarchy: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

        if self.feature_list is None or self.model is None:
            raise NotTrainedException(
                'This classifier is not trained or has not been loaded')
        features_df = features[self.feature_list]
        features_np = self.preprocessor.preprocess_features(
            features_df).values

        predictions = {}
        for classifier_name, classifier in self.model.items():
            probs_np = classifier.predict_proba(features_np)
            df = pd.DataFrame(
                data=probs_np,
                columns=classifier.classes_,
                index=features_df.index
            )
            predictions[classifier_name] = df

        full_probs = pd.concat([
            predictions['top']['transient'].values.reshape(-1, 1) * predictions['transient'],
            predictions['top']['stochastic'].values.reshape(-1, 1) * predictions['stochastic'],
            predictions['top']['periodic'].values.reshape(-1, 1) * predictions['periodic'],
        ], axis=1)

        full_probs = full_probs[self.list_of_classes]
        if return_hierarchy:
            return full_probs, predictions
        else:
            return full_probs

    def fit(
            self,
            features: pd.DataFrame,
            labels: pd.DataFrame,
            config: Dict):
        self.feature_list = features.columns.values
        training_labels = labels[labels['partition'] == 'training_0']
        training_features = features.loc[training_labels.index]
        training_features = self.preprocessor.preprocess_features(
            training_features)

        top_model = BalancedRandomForestClassifier(
            n_estimators=config['n_trees'],
            max_depth=config["max_depth"],
            max_features="sqrt",
            n_jobs=config['n_jobs'],
            verbose=config['verbose']
        )

        stochastic_model = BalancedRandomForestClassifier(
            n_estimators=config['n_trees'],
            max_depth=config["max_depth"],
            max_features="sqrt",
            n_jobs=config['n_jobs'],
            verbose=config['verbose']
        )

        periodic_model = BalancedRandomForestClassifier(
            n_estimators=config['n_trees'],
            max_depth=config["max_depth"],
            max_features="sqrt",
            n_jobs=config['n_jobs'],
            verbose=config['verbose']
        )

        transient_model = BalancedRandomForestClassifier(
            n_estimators=config['n_trees'],
            max_depth=config["max_depth"],
            max_features="sqrt",
            n_jobs=config['n_jobs'],
            verbose=config['verbose']
        )

        transient_mask = training_labels['astro_class'].isin(
            self.class_hierarchy['transient'])
        transient_model.fit(
            training_features[transient_mask].values,
            training_labels['astro_class'][transient_mask].values
        )

        periodic_mask = training_labels['astro_class'].isin(
            self.class_hierarchy['periodic'])
        periodic_model.fit(
            training_features[periodic_mask].values,
            training_labels['astro_class'][periodic_mask].values
        )

        stochastic_mask = training_labels['astro_class'].isin(
            self.class_hierarchy['stochastic'])
        stochastic_model.fit(
            training_features[stochastic_mask].values,
            training_labels['astro_class'][stochastic_mask].values
        )

        inverse_class_hierarchy = {}
        for k, v in self.class_hierarchy.items():
            for astro_class in v:
                inverse_class_hierarchy[astro_class] = k

        top_labels = training_labels['astro_class'].map(
            inverse_class_hierarchy)
        top_model.fit(
            training_features.values,
            top_labels.values
        )

        self.model = {
            'top': top_model,
            'stochastic': stochastic_model,
            'periodic': periodic_model,
            'transient': transient_model
        }

    def save_classifier(self, directory: str):
        if self.model is None:
            raise NotTrainedException(
                'Cannot save model that has not been trained')

        if not os.path.exists(directory):
            os.mkdir(directory)

        with open(
                os.path.join(
                    directory,
                    'hierarchical_random_forest_model.pkl'),
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
        loaded_data = pd.read_pickle(os.path.join(directory, 'hierarchical_random_forest_model.pkl'))
        self.feature_list = loaded_data['feature_list']
        self.list_of_classes = loaded_data['list_of_classes']
        self.model = loaded_data['model']
