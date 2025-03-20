import os
import numpy as np
import pandas as pd
import time

from utils import ZTF_ff_columns_to_PROD
from model import HierarchicalRandomForestClassifier

datasets_order_classes = {
    "ztf_ff": [
        "SNIa", "SNIbc", "SNIIb", "SNII", "SNIIn", "SLSN", 
        "TDE", "Microlensing", "QSO", "AGN", "Blazar", 
        "YSO", "CV/Nova", "LPV", "EA", "EB/EW", 
        "Periodic-Other", "RSCVn", "CEP", "RRLab", "RRLc", "DSCT"
    ],
    "ztf_ff_sanchez_tax": [
        "SNIa", "SNIbc", "SNII", "SLSN",
        "QSO", "AGN", "Blazar",
        "YSO", "CV/Nova", "LPV", "E",
        "DSCT", "RRL", "CEP", "Periodic-Other",
    ],
}

class_hierarchy = {
    'ztf_ff': {
        "Transient": ["SNIa", "SNIbc", "SNIIb", "SNII", "SNIIn", "SLSN", "TDE"],
        "Periodic": [
            "LPV",
            "EA",
            "EB/EW",
            "Periodic-Other",
            "RSCVn",
            "CEP",
            "RRLab",
            "RRLc",
            "DSCT",
        ],
        "Stochastic": [
            "QSO",
            "AGN",
            "Blazar",
            "YSO",
            "CV/Nova",
            "Microlensing",
        ],
    },

    'ztf_ff_sanchez_tax': { 
        "Transient": ["SNIa", "SNIbc", "SNII", "SLSN"],
        "Periodic": [
            "LPV",
            "E",
            "Periodic-Other",
            "CEP",
            "RRL",
            "DSCT",
        ],
        "Stochastic": [
            "QSO",
            "AGN",
            "Blazar",
            "YSO",
            "CV/Nova",
        ],
    }
}


def load_data(path_features, path_partition, path_objects):
    features = pd.read_parquet(path_features)
    partitions = pd.read_parquet(path_partition)
    objects = pd.read_parquet(path_objects).reset_index()  # to get RA/DEC
    return features, partitions, objects


def manage_bias_periodic_others(features, objects, partitions, fold):
    training_partition = f"training_{fold}"
    po_tp = partitions[
        (partitions["class_name"] == "Periodic-Other")
        & (partitions["partition"] == training_partition)
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

def prepare_partitions(labels, label_suffixes):
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

import glob 

def run(path_features, path_partition, path_objects, dir_save_results):

    partition_name = path_partition.split('/')[-2]
    os.makedirs(dir_save_results, exist_ok=True)

    path_results = glob.glob(f'{dir_save_results}/fold_*')

    for path_result in path_results:
        fold = path_result.split('/')[-1].split('_')[-1]
        print(f"Running {path_result} - fold {fold}...")

        features, partitions, objects = load_data(path_features, path_partition, path_objects)

        columns_to_select = list(ZTF_ff_columns_to_PROD.keys()) + ['shorten']
        features = features[columns_to_select]
        features.index = 'aid_' + features.index.astype(str)
        features.index.name = 'index'

        partitions.index = 'aid_' + partitions['oid'].astype(str)

        if 'sanchez' in partition_name:
            name_dataset = 'ztf_ff_sanchez_tax'
        else:
            name_dataset = 'ztf_ff'

        # Manage bias periodic others
        features = manage_bias_periodic_others(features, objects, partitions, fold)

        # support for shortened lightcurves
        features.index = (
            features.index.values + "_" + features["shorten"].astype(str)
        ).values

        list_ndays = features["shorten"].astype(str).unique()
        shorten = features["shorten"]
        features = features[[c for c in features.columns if c != "shorten"]]
        
        partitions = prepare_partitions(partitions, list_ndays)
        list_of_classes = partitions["astro_class"].unique()
        list_of_classes.sort()

        # Model construction
        classifier = HierarchicalRandomForestClassifier(list_of_classes, class_hierarchy[name_dataset])
        sampling_strategy = {
            "Top": dict(zip(["Transient", "Periodic", "Stochastic"], [1_000 * len(list_ndays)] * 3)),
            "Transient": "auto",
            "Periodic": dict(zip(
                classifier.class_hierarchy["Periodic"],
                [200 * len(list_ndays)] * len(classifier.class_hierarchy["Periodic"]),
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

        # Train and evaluate
        evaluate(classifier, features, shorten, partitions, config, path_result, fold)


def evaluate(
        classifier, 
        features, 
        shorten, 
        partitions, 
        config, 
        path_result, 
        fold
        ):

    train_partition = partitions[partitions["partition"] == f"training_{fold}"]
    val_partition = partitions[partitions["partition"] == f"validation_{fold}"]
    train_features = features.loc[train_partition.index]
    val_features = features.loc[val_partition.index]

    """Trains and evaluates a classifier, saving the model."""
    classifier.load_classifier(f'{path_result}/model/hierarchical_random_forest_model.pkl')

    test_labels = partitions[partitions["partition"] == "test"]
    test_features = features.loc[test_labels.index]
    test_probs = classifier.classify_batch(test_features)

    # Compute accuracy per class
    results = pd.concat([test_labels, test_probs.idxmax(axis=1)], axis=1)
    acc = results.groupby("astro_class").apply(lambda x: (x["astro_class"] == x[0]).astype(float).mean())
    print(f"Fold {fold} - Accuracy:\n{acc}")

    # Save the model
    #os.makedirs(os.path.join(dir_save_results, f"fold_{fold}"))
    #classifier.save_classifier(os.path.join(dir_save_results, f"fold_{fold}", 'model'))

    # Save predictions
    test_probs['y_pred'] = test_probs.idxmax(axis=1)
    test_probs['y_true'] = test_labels.astro_class
    test_probs['shorten'] = shorten
    test_probs.to_parquet(os.path.join(path_result, f"predictions_test_a.parquet"))



if __name__ == "__main__":
    path_features = "data_241209_ndetge8_ao_shorten_features/consolidated_features.parquet"
    #path_features = "consolidated_features_16.parquet"
    path_partition = "../../../data_acquisition/ztf_forced_photometry/preprocessed/partitions/241209_ndetge8/partitions.parquet"
    path_objects = "../../../data_acquisition/ztf_forced_photometry/raw/objects.parquet"

    dir_save_results = "results/241209_ndetge8/HBRF_20250219-005312"
    run(
        path_features,
        path_partition,
        path_objects,
        dir_save_results
        )
