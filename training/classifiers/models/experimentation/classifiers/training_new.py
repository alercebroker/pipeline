import os
import numpy as np
import pandas as pd
import time
from alerce_classifiers.classifiers.mlp import MLPClassifier
from alerce_classifiers.classifiers.random_forest import RandomForestClassifier
from alerce_classifiers.classifiers.hierarchical_random_forest import (
    HierarchicalRandomForestClassifier,
)
from alerce_classifiers.classifiers.lightgbm import LightGBMClassifier
from alerce_classifiers.classifiers.xgboost import XGBoostClassifier
from utils import ZTF_ff_columns_to_PROD

def load_data(path_features, path_partition, path_objects):
    features = pd.read_parquet(
        "data_241209_ao_shorten_features/consolidated_features.parquet"
        )
    partitions = pd.read_parquet(
        "../../../data_acquisition/ztf_forced_photometry/preprocessed/partitions/241209/partitions.parquet"
        )
    objects = pd.read_parquet(
        "../../../data_acquisition/ztf_forced_photometry/raw/objects_with_wise_20240105.parquet"
    )  # to get RA/DEC
    return features, partitions, objects


def manage_bias_periodic_others(features, objects, fold):
    training_partition = f"training_{fold}"
    po_tp = labels[
        (labels["class_name"] == "Periodic-Other")
        & (labels["partition"] == training_partition)
    ]
    po_tp_coords = objects[objects["oid"].isin(po_tp["oid"])]
    southern = po_tp_coords["dec"] < -20

    n_not_to_be_replaced = ((~southern).astype(float).sum()) / (54 - (-20)) * (28 - 20)
    n_not_to_be_replaced = int(np.ceil(n_not_to_be_replaced))

    southern_po_oids = po_tp_coords[southern]["oid"].values
    northern_po_oids = po_tp_coords[~southern]["oid"].values

    np.random.seed(0)
    southern_not_to_be_replaced = np.random.choice(
        southern_po_oids, size=n_not_to_be_replaced, replace=False
    )
    not_to_be_replaced = np.concatenate([northern_po_oids, southern_not_to_be_replaced])
    to_be_replaced = list(set(po_tp.oid.values) - set(not_to_be_replaced))

    features.reset_index(inplace=True)
    tbr_feature_mask = features["index"].isin(["aid_" + oid for oid in to_be_replaced])
    n_replacement_needed = tbr_feature_mask.astype(int).sum()
    replacement_coords = (
        features[features["index"].isin(["aid_" + oid for oid in not_to_be_replaced])][
            [f"Coordinate_{x}_nan" for x in "xyz"]
        ]
        .sample(n_replacement_needed, replace=True)
        .values
    )
    features.set_index("index", inplace=True)
    features.loc[
        ["aid_" + oid for oid in to_be_replaced], [f"Coordinate_{x}_nan" for x in "xyz"]
    ] = replacement_coords

    return features


def run(folds_to_run):

    features, partitions, objects = load_data(path_features, path_partition, path_objects)

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    for fold_id in folds_to_run:
        print(f"Running fold {fold_id}...")
        training_partition = f"training_{fold_id}"
        validation_partition = f"validation_{fold_id}"


        columns_to_select = list(ZTF_ff_columns_to_PROD.keys()) + ['shorten']
        features = features[columns_to_select]
        features.index = 'aid_' + features.index.astype(str)
        features.index.name = 'index'


        features, list_ndays = manage_bias_periodic_others(fold_id, objects)

        # support for shortened lightcurves
        features.index = (
            features.index.values + "_" + features["shorten"].astype(str)
        ).values

        list_ndays = features["shorten"].astype(str).unique()
        features = features[[c for c in features.columns if c != "shorten"]]


        labels = prepare_labels(labels, list_ndays)
        list_of_classes = labels["astro_class"].unique()
        list_of_classes.sort()




    # Loop through selected folds
    for fold_id in folds_to_run:
        print(f"Running fold {fold_id}...")
        partition_name = f"training_{fold_id}"

        # Filter features and labels for the current fold
        train_labels = labels[labels["partition"] == partition_name]
        val_labels = labels[labels["partition"] == f"validation_{fold_id}"]

        train_features = features.loc[train_labels.index]
        val_features = features.loc[val_labels.index]

        # Initialize classifier
        classifier_type = "HierarchicalRandomForest"  # Change as needed
        if classifier_type == "HierarchicalRandomForest":
            classifier = HierarchicalRandomForestClassifier(list_of_classes)
            sampling_strategy = {
                "Top": dict(zip(["Transient", "Periodic", "Stochastic"], [1_000 * len(label_suffixes)] * 3)),
                "Transient": "auto",
                "Periodic": dict(zip(
                    classifier.class_hierarchy["Periodic"],
                    [200 * len(label_suffixes)] * len(classifier.class_hierarchy["Periodic"]),
                )),
                "Stochastic": "auto",
            }
            config = {
                "n_trees": 500,
                "max_depth": 10,
                "n_jobs": 8,
                "verbose": 11,
                "sampling": sampling_strategy,
            }

        train_and_evaluate(classifier, train_features, train_labels, config, partition_name, models_dir, fold_id)


def prepare_labels(labels, label_suffixes):
    """Prepares labels by handling suffixes and renaming columns."""
    label_list = []
    for suffix in label_suffixes:
        labels_copy = labels.copy()
        labels_copy["aid"] = "aid_" + labels_copy["oid"] + "_" + suffix
        label_list.append(labels_copy)

    labels = pd.concat(label_list, axis=0)
    labels.rename(columns={"class_name": "astro_class"}, inplace=True)
    labels.set_index("aid", inplace=True)
    return labels

def train_and_evaluate(classifier, features, labels, config, partition_name, models_dir, fold_id):
    """Trains and evaluates a classifier, saving the model."""
    classifier.fit(features, labels, config)

    test_labels = labels[labels["partition"] == "test"]
    test_features = features.loc[test_labels.index]
    test_probs = classifier.classify_batch(test_features)

    # Compute accuracy per class
    results = pd.concat([test_labels, test_probs.idxmax(axis=1)], axis=1)
    acc = results.groupby("astro_class").apply(lambda x: (x["astro_class"] == x[0]).astype(float).mean())
    print(f"Fold {fold_id} - Accuracy:\n{acc}")

    # Save the model
    classifier.save_classifier(os.path.join(models_dir, f"{partition_name}_fold{fold_id}.pkl"))



if __name__ == "__main__":
    folds_to_run = [0, 1, 2, 3, 4]  # Specify which folds to run (e.g., [0] for a single fold)
    run(folds_to_run)

    ##########

    ##########

    ################################################################################
    # manage bias of Periodic-Other towards southern sky


    # rename features to match avro schema
    # features.rename(columns=rename_feature, inplace=True)
    ################################################################################

    label_list = []
    for suffix in label_suffixes:
        labels_copy = labels.copy()
        # labels_copy['aid'] += '_' + suffix
        labels_copy["aid"] = "aid_" + labels_copy["oid"] + "_" + suffix
        label_list.append(labels_copy)

    labels = pd.concat(label_list, axis=0)

    labels.rename(columns={"class_name": "astro_class"}, inplace=True)
    labels.set_index("aid", inplace=True)
    list_of_classes = labels["astro_class"].unique()
    list_of_classes.sort()

    classifier_type = "HierarchicalRandomForest"
    models_dir = "models"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(f'timestr: {timestr}')

    if classifier_type == "MLP":
        ztf_classifier = MLPClassifier(list_of_classes)
        config = {"learning_rate": 1e-4, "batch_size": 4096}
        ztf_classifier.fit(features, labels, config)
        ztf_classifier.save_classifier(os.path.join(models_dir, f"mlp_{timestr}"))
    elif classifier_type == "RandomForest":
        classifier = RandomForestClassifier(list_of_classes)
        config = {"n_trees": 500, "n_jobs": 8, "verbose": 11}
        classifier.fit_from_features(features, labels, config)

        test_labels = labels[labels["partition"] == "test"]
        test_features = features.loc[test_labels["aid"].values]

        test_probs = classifier.classify_batch_from_features(test_features)
        test_labels.set_index("aid", inplace=True)
        a = pd.concat([test_labels, test_probs.idxmax(axis=1)], axis=1)
        acc = a.groupby("astro_class").apply(
            lambda x: (x["astro_class"] == x[0]).astype(float).mean()
        )
        print(acc)

        classifier.save_classifier("rf_classifier_240307")
    elif classifier_type == "HierarchicalRandomForest":
        classifier = HierarchicalRandomForestClassifier(list_of_classes)

        sampling_strategy = {
            "Top": dict(
                zip(
                    ["Transient", "Periodic", "Stochastic"],
                    [1_000 * len(label_suffixes)] * 3,
                )
            ),
            "Transient": "auto",
            "Periodic": dict(
                zip(
                    classifier.class_hierarchy["Periodic"],
                    [200 * len(label_suffixes)]
                    * len(classifier.class_hierarchy["Periodic"]),
                )
            ),
            "Stochastic": "auto",
        }

        config = {
            "n_trees": 500,
            "max_depth": 10,
            "n_jobs": 8,
            "verbose": 11,
            "sampling": sampling_strategy,
        }
        classifier.fit(features, labels, config)

        test_labels = labels[labels["partition"] == "test"]
        test_features = features.loc[test_labels.index]

        test_probs = classifier.classify_batch(test_features)
        a = pd.concat([test_labels, test_probs.idxmax(axis=1)], axis=1)
        acc = a.groupby("astro_class").apply(
            lambda x: (x["astro_class"] == x[0]).astype(float).mean()
        )
        print(acc)
        classifier.save_classifier(
            os.path.join(models_dir, f"hrf_classifier_{timestr}")
        )
    elif classifier_type == "LightGBM":
        classifier = LightGBMClassifier(list_of_classes)
        config = {}
        classifier.fit_from_features(features, labels, config)

        test_labels = labels[labels["partition"] == "test"]
        test_features = features.loc[test_labels["aid"].values]

        test_probs = classifier.classify_batch_from_features(test_features)
        test_labels.set_index("aid", inplace=True)
        a = pd.concat([test_labels, test_probs.idxmax(axis=1)], axis=1)
        acc = a.groupby("astro_class").apply(
            lambda x: (x["astro_class"] == x[0]).astype(float).mean()
        )
        print(acc)
        classifier.save_classifier("lightgbm_classifier_240311")
    elif classifier_type == "XGBoost":
        classifier = XGBoostClassifier(list_of_classes)
        config = {}
        classifier.fit_from_features(features, labels, config)

        test_labels = labels[labels["partition"] == "test"]
        test_features = features.loc[test_labels["aid"].values]

        test_probs = classifier.classify_batch_from_features(test_features)
        test_labels.set_index("aid", inplace=True)
        a = pd.concat([test_labels, test_probs.idxmax(axis=1)], axis=1)
        acc = a.groupby("astro_class").apply(
            lambda x: (x["astro_class"] == x[0]).astype(float).mean()
        )
        print(acc)
        classifier.save_classifier("xgboost_classifier_240312")
