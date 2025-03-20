from typing import List, Dict, Union, Tuple

import pandas as pd
import numpy as np
import os
import pickle

from imblearn.ensemble import BalancedRandomForestClassifier


class HierarchicalRandomForestClassifier:
    def __init__(self, list_of_classes: List[str], class_hierarchy: dict, name_dataset: str):
        self.class_hierarchy = class_hierarchy
        self.list_of_classes = list_of_classes
        self.name_dataset = name_dataset
        self.dict_of_rf = None
        self.feature_list = None

    def classify_batch(
        self, features: pd.DataFrame, return_hierarchy: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:

        if self.feature_list is None or self.dict_of_rf is None:
            raise "This classifier is not trained or has not been loaded"

        features = self.preprocess_features(features)
        features_np = features[self.feature_list].values

        predictions = {}
        for classifier_name, classifier in self.dict_of_rf.items():
            probs_np = classifier.predict_proba(features_np)
            df = pd.DataFrame(
                data=probs_np, columns=classifier.classes_, index=features.index
            )
            predictions[classifier_name] = df

        full_probs = pd.concat(
            [
                predictions["top"]["Transient"].values.reshape(-1, 1)
                * predictions["Transient"],
                predictions["top"]["Stochastic"].values.reshape(-1, 1)
                * predictions["Stochastic"],
                predictions["top"]["Periodic"].values.reshape(-1, 1)
                * predictions["Periodic"],
            ],
            axis=1,
        )

        # Solo agregar "Others" si el dataset contiene 'sanchez'
        if 'sanchez' in self.name_dataset and "Others" in predictions["top"].columns:
            others_probs = predictions["top"]["Others"].values.reshape(-1, 1)
            others_df = pd.DataFrame(others_probs, index=features.index, columns=["Others"])
            full_probs = pd.concat([full_probs, others_df], axis=1)

        full_probs = full_probs[self.list_of_classes]
        if return_hierarchy:
            return full_probs, predictions
        else:
            return full_probs

    def fit(self, train_features: pd.DataFrame, train_partition: pd.DataFrame, config: Dict):
        train_features = self.preprocess_features(train_features)
        self.feature_list = train_features.columns.values

        top_model = BalancedRandomForestClassifier(
            n_estimators=config["n_trees"],
            max_depth=config["max_depth"],
            sampling_strategy=config["sampling"]["Top"],
            max_features="sqrt",
            n_jobs=config["n_jobs"],
            verbose=config["verbose"],
        )

        stochastic_model = BalancedRandomForestClassifier(
            n_estimators=config["n_trees"],
            max_depth=config["max_depth"],
            sampling_strategy=config["sampling"]["Stochastic"],
            max_features="sqrt",
            n_jobs=config["n_jobs"],
            verbose=config["verbose"],
        )

        periodic_model = BalancedRandomForestClassifier(
            n_estimators=config["n_trees"],
            max_depth=config["max_depth"],
            sampling_strategy=config["sampling"]["Periodic"],
            max_features="sqrt",
            n_jobs=config["n_jobs"],
            verbose=config["verbose"],
        )

        transient_model = BalancedRandomForestClassifier(
            n_estimators=config["n_trees"],
            max_depth=config["max_depth"],
            sampling_strategy=config["sampling"]["Transient"],
            max_features="sqrt",
            n_jobs=config["n_jobs"],
            verbose=config["verbose"],
        )

        transient_mask = train_partition["astro_class"].isin(
            self.class_hierarchy["Transient"]
        )
        transient_model.fit(
            train_features[transient_mask].values,
            train_partition["astro_class"][transient_mask].values,
        )

        periodic_mask = train_partition["astro_class"].isin(
            self.class_hierarchy["Periodic"]
        )
        periodic_model.fit(
            train_features[periodic_mask].values,
            train_partition["astro_class"][periodic_mask].values,
        )

        stochastic_mask = train_partition["astro_class"].isin(
            self.class_hierarchy["Stochastic"]
        )
        stochastic_model.fit(
            train_features[stochastic_mask].values,
            train_partition["astro_class"][stochastic_mask].values,
        )

        inverse_class_hierarchy = {}
        for k, v in self.class_hierarchy.items():
            for astro_class in v:
                inverse_class_hierarchy[astro_class] = k

        top_labels = train_partition["astro_class"].map(inverse_class_hierarchy)

        # Incluir "Others" como una clase independiente en el modelo "top"
        if 'sanchez' in self.name_dataset: 
            top_labels.loc[train_partition["astro_class"] == "Others"] = "Others"

        top_model.fit(train_features.values, top_labels.values)

        self.dict_of_rf = {
            "top": top_model,
            "Stochastic": stochastic_model,
            "Periodic": periodic_model,
            "Transient": transient_model,
        }

    def save_classifier(self, directory: str):
        if self.dict_of_rf is None:
            raise "Cannot save model that has not been trained"

        if not os.path.exists(directory):
            os.mkdir(directory)

        with open(
            os.path.join(directory, "hierarchical_random_forest_model.pkl"), "wb"
        ) as f:
            pickle.dump(
                {
                    "feature_list": self.feature_list,
                    "list_of_classes": self.list_of_classes,
                    "model": self.dict_of_rf,
                },
                f,
            )

    def load_classifier(self, model_path: str, n_jobs: int = 1, verbose: int = 0):
        if model_path.split(".")[-1] == "pkl":
            filename = model_path
        else:
            filename = os.path.join(model_path, "hierarchical_random_forest_model.pkl")
        loaded_data = pd.read_pickle(filename)
        self.feature_list = loaded_data["feature_list"]
        self.list_of_classes = loaded_data["list_of_classes"]
        self.dict_of_rf = loaded_data["model"]
        for key, rf in self.dict_of_rf.items():
            rf.n_jobs = n_jobs
            rf.verbose = verbose

    def preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        features = inf_to_nan(features.astype(np.float64)).copy()
        features = features.fillna(-999.0)
        features.columns = self._preprocess_feature_names(features.columns.values)
        return features

    def _preprocess_feature_names(self, feature_names: List[str]):
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


def inf_to_nan(features: pd.DataFrame) -> pd.DataFrame:
    features = features.replace([np.inf, -np.inf], np.nan)
    return features