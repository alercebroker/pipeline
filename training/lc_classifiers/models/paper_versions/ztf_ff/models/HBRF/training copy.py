import multiprocessing as mp
import pandas as pd
import numpy as np
import itertools
import glob
import yaml
import time
import sys
import os

# Agregar la ruta al directorio que contiene 'schema.py'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tqdm import tqdm

from utils.features_processing import process_batch
from model import HierarchicalRandomForestClassifier
from schema import ZTF_ff_columns_to_PROD

datasets_order_classes = {
    "ztf_ff": [
        "SNIa", "SESN", "SNII", "SNIIn", "SLSN",  # , "SNIIb"
        "TDE", "Microlensing", "QSO", "AGN", "Blazar", 
        "YSO", "CV/Nova", "LPV", "EA", "EB/EW", 
        "Periodic-Other", "RSCVn", "CEP", "RRLab", "RRLc", "DSCT"
    ],
    "ztf_ff_sanchez_tax": [
        "SNIa", "SNIbc", "SNII", "SLSN",
        "QSO", "AGN", "Blazar",
        "YSO", "CV/Nova", "LPV", "E",
        "DSCT", "RRL", "CEP", "Periodic-Other", "Others"
    ],
}

class_hierarchy = {
    'ztf_ff': {
        "Transient": ["SNIa", "SESN", "SNII", "SNIIn", "SLSN", "TDE"], #, "SNIIb"
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


def save_hyperparams(config, trial_dir):
    """Guarda los hiperparámetros de un trial en un archivo YAML."""
    hyperparam_path = os.path.join(trial_dir, "hparams.yaml")
    with open(hyperparam_path, 'w') as file:
        yaml.dump(config, file, sort_keys=False)


def consolidate_features(path_features, num_data_loading_workers):
    #dir_name = path_features.split('/')[-1]
    data_dir = glob.glob(f'{path_features}/*')
    data_dir = [filename for filename in data_dir if "astro_objects_batch" in filename]
    data_dir = sorted(data_dir)
    
    num_workers = min(mp.cpu_count(), num_data_loading_workers)  # Use available CPU cores
    with mp.Pool(processes=num_workers) as pool:
        all_features = list(tqdm(pool.imap(process_batch, data_dir), total=len(data_dir)))
    
    all_features = pd.concat(all_features, axis=0)
    object_columns = all_features.select_dtypes(include=["object"]).columns
    columns_to_convert = object_columns.drop("shorten", errors="ignore")
    all_features[columns_to_convert] = all_features[columns_to_convert].apply(pd.to_numeric, errors="coerce")

    return all_features


def load_data(path_features, path_partition, path_objects, num_data_loading_workers):
    features = consolidate_features(path_features, num_data_loading_workers)
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
            [f"Coordinate_{x}" for x in "xyz"]
        ]
        .sample(n_replacement_needed, replace=True)
        .values
    )
    features.set_index("index", inplace=True)
    features.loc[
        ["aid_" + oid for oid in to_be_replaced], [f"Coordinate_{x}" for x in "xyz"]
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

def run_with_hyperparams(
        path_features, 
        path_partition, 
        path_objects, 
        folds_to_run, 
        param_grid, 
        rm_feat, 
        num_data_loading_workers, 
        searching_hyperparameters
        ):

    models_dir = "results"
    partition_name = path_partition.split('/')[-2]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dir_save_results = os.path.join(models_dir, partition_name, f"HBRF_{timestr}")
    os.makedirs(dir_save_results, exist_ok=True)

    print(f"Loading data...")
    features, partitions, objects = load_data(path_features, path_partition, path_objects, num_data_loading_workers)

    # Crear todas las combinaciones de hiperparámetros
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for trial_idx, config in enumerate(param_combinations):
        print(f"Trying trial {trial_idx}: {config}\n")

        if searching_hyperparameters:
            trial_dir = os.path.join(dir_save_results, f"trial_{trial_idx}")
        else:
            trial_dir = os.path.join(dir_save_results)
        os.makedirs(trial_dir, exist_ok=True)

        save_hyperparams(config, trial_dir)

        for fold in folds_to_run:
            print(f"Running fold {fold}...")
            all_columns = sorted(ZTF_ff_columns_to_PROD.keys())
            columns_to_select = [col for col in all_columns if not any(substring in col for substring in rm_feat)] + ['shorten']

            fold_features = features[columns_to_select].copy()
            fold_features.index = 'aid_' + fold_features.index.astype(str)
            fold_features.index.name = 'index'

            fold_partitions = partitions.copy()
            fold_partitions.index = 'aid_' + fold_partitions['oid'].astype(str)

            if 'sanchez' in partition_name:
                name_dataset = 'ztf_ff_sanchez_tax'
            else:
                name_dataset = 'ztf_ff'

            # Manage bias periodic others
            fold_features = manage_bias_periodic_others(fold_features, objects, fold_partitions, fold)

            # support for shortened lightcurves
            fold_features.index = (
                fold_features.index.values + "_" + fold_features["shorten"].astype(str)
            ).values

            list_ndays = fold_features["shorten"].astype(str).unique()
            shorten = fold_features["shorten"]
            fold_features = fold_features[[c for c in fold_features.columns if c != "shorten"]]

            fold_partitions = prepare_partitions(fold_partitions, list_ndays)
            list_of_classes = fold_partitions["astro_class"].unique()
            list_of_classes.sort()

            # Model construction
            classifier = HierarchicalRandomForestClassifier(list_of_classes, class_hierarchy[name_dataset], name_dataset)
            sampling_strategy = {
                "Top": dict(zip(["Transient", "Periodic", "Stochastic"], [1_000 * len(list_ndays)] * 3)),
                "Transient": "auto",
                "Periodic": dict(zip(
                    classifier.class_hierarchy["Periodic"],
                    [200 * len(list_ndays)] * len(classifier.class_hierarchy["Periodic"]),
                )),
                "Stochastic": "auto",
            }
            config['sampling'] = sampling_strategy

            train_and_evaluate(classifier, fold_features, shorten, fold_partitions, config, trial_dir, fold, searching_hyperparameters, name_dataset)


def train_and_evaluate(
        classifier, 
        features, 
        shorten, 
        partitions, 
        config, 
        trial_dir, 
        fold,
        searching_hyperparameters,
        name_dataset
        ):

    train_partition = partitions[partitions["partition"] == f"training_{fold}"]
    train_features = features.loc[train_partition.index]

    if searching_hyperparameters:
        eval_partition = partitions[partitions["partition"] == f"validation_{fold}"]
        print(f"Evaluating on validation set for fold {fold}")
    else:
        eval_partition = partitions[partitions["partition"] == "test"]
        print(f"Evaluating on test set for fold {fold}")

    eval_features = features.loc[eval_partition.index]

    # Train
    classifier.fit(train_features, train_partition, config)

    # Predict
    eval_probs, hier_probs = classifier.classify_batch(eval_features, return_hierarchy=True)

    # Compute accuracy per class
    results = pd.concat([eval_partition, eval_probs.idxmax(axis=1)], axis=1)
    acc = results.groupby("astro_class").apply(lambda x: (x["astro_class"] == x[0]).astype(float).mean())
    print(f"Fold {fold} - Accuracy:\n{acc}")

    # Save model only if it's the final test (not during hyperparameter tuning)
    path_save = os.path.join(trial_dir, f"fold_{fold}")
    os.makedirs(path_save, exist_ok=True)

    if not searching_hyperparameters:
        classifier.save_classifier(os.path.join(path_save, 'model'))

    # Save predictions
    eval_probs['y_pred'] = eval_probs.idxmax(axis=1)
    eval_probs['y_true'] = eval_partition.astro_class
    eval_probs['shorten'] = shorten
    suffix = "validation" if searching_hyperparameters else "test"
    eval_probs.to_parquet(os.path.join(path_save, f"predictions_{suffix}.parquet"))
    
    if name_dataset == 'ztf_ff' and not searching_hyperparameters:
        # Save Tops predictions
        hier_probs['top']['y_pred'] = hier_probs['top'].idxmax(axis=1)

        ztf_ff_flat = {subclass: superclass for superclass, subclasses in class_hierarchy[name_dataset].items() for subclass in subclasses}
        eval_partition["astro_class"] = eval_partition["astro_class"].replace(ztf_ff_flat)
        hier_probs['top']['y_true'] = eval_partition.astro_class

        hier_probs['top']['shorten'] = shorten
        hier_probs['top'].to_parquet(os.path.join(path_save, f"predictions_{suffix}_top.parquet"))


if __name__ == "__main__":
    name_data_version = "250408_ndetge8"
    path_features = f"../../data/preprocessed/data_{name_data_version}_ao_shorten_features"
    path_objects = "../../data/preprocessed/objects_250410.parquet"

    rm_feat = ['TDE_mag0', 'fleet_t0', 'fleet_m0', 'ulens_t0', 'ulens_mag0']
    num_data_loading_workers = 20
    searching_hyperparameters = False

    if searching_hyperparameters:
        path_partition = f"../../data/partitions/{name_data_version}/partitions.parquet"
        folds_to_run = [0, 1, 2, 3, 4]
        param_grid = {
            "n_trees": [10, 100, 350, 500],  # Número de árboles en el bosque
            "criterion": ["gini", "entropy"],  # Criterio de división
            "max_depth": [10, 100, None],  # Profundidad máxima del árbol (None significa sin límite)
            "n_jobs": [8],
            "verbose": [11],
        }

    else:
        #path_partition = f"../../data/partitions/{name_data_version}_20folds/partitions.parquet"
        path_partition = f"../../data/partitions/{name_data_version}_sanchez_tax_20folds/partitions.parquet"
        #path_partition = f"../../data/partitions/{name_data_version}_grouped_tax_20folds/partitions.parquet"
        folds_to_run = range(20)
        param_grid = {
            "n_trees": [500],  # Número de árboles en el bosque
            "criterion": ["gini"],  # Criterio de división
            "max_depth": [100],  # Profundidad máxima del árbol (None significa sin límite)
            "n_jobs": [8],
            "verbose": [11],
        }

    run_with_hyperparams(
        path_features,
        path_partition,
        path_objects,
        folds_to_run,
        param_grid,
        rm_feat,
        num_data_loading_workers,
        searching_hyperparameters
        )
