import pandas as pd
import numpy as np

import warnings
import json
import os

from sklearn.model_selection import train_test_split, StratifiedKFold

warnings.filterwarnings("ignore")


def ordered_partitions(df_lc, partitions, fold, dict_cols):
    this_partition = partitions[
        (partitions["partition"] == "training_%d" % fold)
        | (partitions["partition"] == "validation_%d" % fold)
        | (partitions["partition"] == "test")
    ]
    this_partition = this_partition.set_index(dict_cols["oid"])
    this_partition_final = this_partition.filter(items=df_lc.index, axis=0)
    this_partition_final["unique_id"] = np.arange(len(this_partition_final))
    this_partition_final = this_partition_final.reset_index().rename(
        columns={"index": dict_cols["oid"]}
    )
    this_partition_final = this_partition_final.set_index("unique_id")
    return this_partition_final


def open_partitions(partitions, dict_cols):
    mapping_to_int = {
        key: idx for idx, key in enumerate(partitions[dict_cols["class"]].unique())
    }

    def apply_mapping(label_str):
        return mapping_to_int[label_str]

    partitions["label_int"] = partitions.apply(
        lambda x: apply_mapping(x[dict_cols["class"]]), axis=1
    )
    return partitions, mapping_to_int


def get_partitions(df_objid_label, dict_cols, path_save_k_fold):
    if not os.path.exists("{}/partitions.parquet".format(path_save_k_fold)):

        X = df_objid_label[dict_cols["oid"]]
        y = df_objid_label[dict_cols["class"]]

        # Dividir en conjunto de prueba y otro (entrenamiento + validación)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Obtener los índices del conjunto de prueba
        test_indices = X_test.index
        test_set = df_objid_label.iloc[test_indices]
        test_set["partition"] = "test"

        training_validation = df_objid_label.drop(test_indices)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        training_set_list = []
        validation_set_list = []
        n_samples = len(training_validation)
        for i, indexes in enumerate(
            kf.split(np.zeros(n_samples), training_validation[dict_cols["class"]])
        ):
            training_index, validation_index = indexes
            training_set = training_validation.iloc[training_index].copy()
            validation_set = training_validation.iloc[validation_index].copy()
            training_set["partition"] = f"training_{i}"
            validation_set["partition"] = f"validation_{i}"
            training_set_list.append(training_set)
            validation_set_list.append(validation_set)

        partitions = pd.concat(
            [test_set] + training_set_list + validation_set_list, axis=0
        )

        blind_test = test_set.copy()
        blind_test[dict_cols["class"]] = "unknown"
        blind_partitions = pd.concat(
            [blind_test] + training_set_list + validation_set_list, axis=0
        )

        os.makedirs(path_save_k_fold, exist_ok=True)
        partitions.to_parquet("{}/partitions.parquet".format(path_save_k_fold))
        blind_partitions.to_parquet(
            "{}/blind_partitions.parquet".format(path_save_k_fold)
        )
        print("The partitions were created.")

    else:
        partitions = pd.read_parquet("{}/partitions.parquet".format(path_save_k_fold))
        blind_partitions = pd.read_parquet(
            "{}/blind_partitions.parquet".format(path_save_k_fold)
        )
        print("We are using partitions that had already been created.")

    partitions, mapping_to_int = open_partitions(partitions, dict_cols)

    return partitions, mapping_to_int
