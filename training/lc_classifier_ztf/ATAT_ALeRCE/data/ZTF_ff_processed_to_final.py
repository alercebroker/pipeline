import pandas as pd
import numpy as np

import yaml
import h5py
import glob
import copy
import os

from sklearn.preprocessing import QuantileTransformer
from joblib import dump

from src.partitions import get_partitions, ordered_partitions
from src.processing import processing_lc
from src.create_dataset import create_lc_h5py
from src.add_md_feat import add_metadata, add_features, compute_feature_quantiles


def check_files(df_objid_label, dict_cols, dict_info, path_lcs_file, path_md_feat_file):
    print("We are checking the IDs in all files that you will use...")
    oid_objects = df_objid_label[dict_cols["oid"]].values
    oid_files = []
    path_lcs_chunks = glob.glob("{}/lightcurves*".format(path_lcs_file))
    for i, path_chunk in enumerate(path_lcs_chunks):
        # print('Checking chunk {}'.format(i))
        num_batch = path_chunk.split("_")[-1].split(".")[0]
        oid_lcs = pd.read_parquet("{}".format(path_chunk))[dict_cols["oid"]].values

        if os.path.exists("./{}/metadata".format(path_md_feat_file)):
            metadata = pd.read_parquet(
                "{}/metadata/metadata_batch_0{}.parquet".format(
                    path_md_feat_file, num_batch
                )
            )
            oid_mds = metadata[dict_cols["oid"]].values

            oid_lcs = np.intersect1d(oid_lcs, oid_mds)

        if os.path.exists("./{}/features".format(path_md_feat_file)):
            for time_to_eval in dict_info["list_time_to_eval"]:
                features = pd.read_parquet(
                    "{}/features/{}_days/feat_batch_0{}.parquet".format(
                        path_md_feat_file, time_to_eval, num_batch
                    )
                )
                oid_feat = features[dict_cols["oid"]].values
                oid_lcs = np.intersect1d(oid_lcs, oid_feat)

        oid_files.append(oid_lcs)

    oid_files = np.concatenate(oid_files)
    all_oids = np.intersect1d(oid_objects, oid_files)
    return all_oids


def main(
    path_lcs_file,
    path_md_feat_file,
    dict_cols,
    dict_info,
    df_objid_label,
    path_save_dataset,
    path_save_k_fold,
):

    print("Total number of objects: {}".format(len(df_objid_label)))
    df_objid_label = df_objid_label[
        df_objid_label[dict_cols["class"]].isin(dict_info["classes_to_use"])
    ].reset_index(drop=True)
    print(
        "Total number of objects after filtering classes used: {}".format(
            len(df_objid_label)
        )
    )

    # We check to have the IDs in all the files that we will use
    all_oids = check_files(
        df_objid_label, dict_cols, dict_info, path_lcs_file, path_md_feat_file
    )
    df_objid_label = df_objid_label[
        df_objid_label[dict_cols["oid"]].isin(all_oids)
    ].reset_index(drop=True)

    if dict_info["rm_some_ulens"]["rm"]:
        ids_rm_ulens = pd.read_csv(dict_info["rm_some_ulens"]["path_ulens"])[
            "oid"
        ].values
        df_objid_label = df_objid_label[
            ~df_objid_label[dict_cols["oid"]].isin(ids_rm_ulens)
        ].reset_index(drop=True)

    all_oids = df_objid_label[dict_cols["oid"]].values

    print(
        "Total number of objects used to create the dataset: {}".format(len(all_oids))
    )

    # --------------------------------  Generate partitions --------------------------------#
    df_partitions, mapping_to_int = get_partitions(
        df_objid_label, dict_cols, path_save_k_fold
    )

    # --------------------------------  Create dataset.h5 for Light Curves --------------------------------#
    # Revisar el separate_by_filter
    df_lc = processing_lc(path_lcs_file, dict_cols, dict_info, df_objid_label)

    #######################################################################################################################################################

    if dict_info["type_windows"] == "windows":

        print(
            "We are expanding the number of windows per light curve in test partition..."
        )

        name_cols = [
            dict_cols["oid"],
            "window_id",
            dict_cols["class"],
            "partition",
            "label_int",
        ]
        partitions_final = []

        test_partition = df_partitions[df_partitions["partition"] == "test"]
        df_partition_expand = pd.merge(
            test_partition, df_lc, on="oid", how="outer"
        ).dropna()[name_cols]
        df_partition_expand["window_id"] = df_partition_expand["window_id"].astype(int)
        partitions_final.append(df_partition_expand)

        # ids_test = test_partition[dict_cols['oid']].values
        # for snid in ids_test:
        #    expands_ids(partitions_final, snid, test_partition, dict_snid_windows, dict_cols)

        for fold in range(5):
            print(
                "We are expanding the number of windows per light curve in fold {}...".format(
                    fold
                )
            )
            this_partitions = df_partitions[
                (df_partitions["partition"] == "training_%d" % fold)
                | (df_partitions["partition"] == "validation_%d" % fold)
            ]

            df_partition_expand = pd.merge(
                this_partitions, df_lc, on="oid", how="outer"
            ).dropna()[name_cols]
            df_partition_expand["window_id"] = df_partition_expand["window_id"].astype(
                int
            )

            if dict_info["undersampling_windows"]["apply"]:
                if dict_info["undersampling_windows"]["same_windows_by_folds"]:
                    if fold == 0:
                        df_keep_windows = df_partition_expand[
                            df_partition_expand["alerceclass"].isin(
                                dict_info["undersampling_windows"]["keep_classes"]
                            )
                        ]
                        df_partition_expand = df_partition_expand.loc[
                            ~df_partition_expand.index.isin(df_keep_windows.index)
                        ]

                        df_random_window_ids = (
                            df_partition_expand.groupby("oid")
                            .apply(lambda group: group.sample(n=1))
                            .reset_index(drop=True)
                        )
                        df_final_windows = pd.concat(
                            [df_keep_windows, df_random_window_ids]
                        )
                        df_aux = df_final_windows.copy()
                    else:
                        df_final_windows = pd.merge(
                            df_partition_expand,
                            df_aux[["oid", "window_id"]],
                            on=["oid", "window_id"],
                        )

                else:
                    df_keep_windows = df_partition_expand[
                        df_partition_expand["alerceclass"].isin(
                            dict_info["undersampling_windows"]["keep_classes"]
                        )
                    ]
                    df_partition_expand = df_partition_expand.loc[
                        ~df_partition_expand.index.isin(df_keep_windows.index)
                    ]
                    df_random_window_ids = (
                        df_partition_expand.groupby("oid")
                        .apply(lambda group: group.sample(n=1))
                        .reset_index(drop=True)
                    )
                    df_final_windows = pd.concat(
                        [df_keep_windows, df_random_window_ids]
                    )

                partitions_final.append(df_final_windows)

            else:
                partitions_final.append(df_partition_expand)

            # ids_train_val = this_partition[dict_cols['oid']].values
            # for snid in ids_train_val:
            #    expands_ids(partitions_final, snid, this_partition, dict_snid_windows, dict_cols)

        df_partitions = pd.concat(partitions_final)
        df_partitions[dict_cols["oid"]] = (
            df_partitions[dict_cols["oid"]].astype(str)
            + "_"
            + df_partitions["window_id"].astype(str)
        )
        df_partitions = df_partitions.drop(columns=["window_id"])

        df_lc[dict_cols["oid"]] = (
            df_lc[dict_cols["oid"]].astype(str) + "_" + df_lc["window_id"].astype(str)
        )
        df_lc = df_lc.drop(columns=["window_id"])
        df_lc = df_lc[df_lc.oid.isin(df_partitions.oid.unique())]
        df_lc = df_lc.set_index(dict_cols["oid"])

    #######################################################################################################################################################

    # Merge between light curves and partitions

    # we will modify the training data for fold_0,
    # so using other partitions will leak info
    num_folds = 1  # 5
    all_partitions = {}
    for fold in range(num_folds):
        all_partitions["fold_%s" % fold] = ordered_partitions(
            df_lc, df_partitions, fold, dict_cols
        )

    df_lc = pd.merge(
        df_lc,
        all_partitions["fold_0"],
        left_index=True,
        right_on=dict_cols["oid"],
        how="inner",
    )[[dict_cols["oid"]] + df_lc.columns.tolist() + ["label_int"]]

    print("We are creating the LC dataset ...")
    os.makedirs(path_save_dataset, exist_ok=True)
    create_lc_h5py(all_partitions, df_lc, num_folds, path_save_dataset)

    # --------------------------------  Add Metadata to the dataset --------------------------------#
    # Esto deberia abrir el dataset de curvas de luz y agregarlos igual que las features
    md_cols = []
    if os.path.exists("{}/metadata".format(path_md_feat_file)):
        print("Adding metadata to the dataset ...")
        path_md_chunks = glob.glob("{}/metadata/*".format(path_md_feat_file))
        h5_file = h5py.File("{}/dataset.h5".format(path_save_dataset))
        All_SNID = h5_file.get("SNID")[:]
        h5_file.close()

        md_cols = add_metadata(
            All_SNID,
            all_oids,
            path_md_chunks,
            dict_cols,
            dict_info,
            path_save_dataset,
            all_partitions,
            num_folds,
        )

    # --------------------------------  Add Calculated Features to the dataset --------------------------------#
    feat_cols = []
    if os.path.exists("{}/features".format(path_md_feat_file)):
        print("Adding calculated features to the dataset ...")
        path_dataset = "{}/dataset.h5".format(path_save_dataset)
        h5_file = h5py.File(path_dataset)
        All_SNID = h5_file.get("SNID")[:]
        h5_file.close()

        feat_cols = add_features(
            All_SNID,
            all_oids,
            path_md_feat_file,
            dict_cols,
            dict_info,
            path_save_dataset,
            dict_info["list_time_to_eval"],
            all_partitions,
            num_folds,
            df_objid_label
        )

        compute_feature_quantiles(path_dataset, path_save_dataset, dict_info["list_time_to_eval"])

    dict_info.update(
        {"mapping_classes": mapping_to_int, "md_cols": md_cols, "feat_cols": feat_cols,}
    )

    with open("{}/dict_info.yaml".format(path_save_dataset), "w") as f:
        yaml.dump(dict_info, f)


if __name__ == "__main__":
    ROOT = "./data/datasets/ZTF_ff"

    version = "240627"  # Consider the partition version that you want to use
    path_save_dataset = "{}/final/LC_MD_FEAT_{}".format(ROOT, version)
    path_save_k_fold = "{}/partitions/{}".format(ROOT, version)

    path_lcs_file = "{}/raw/data_231206".format(ROOT)
    path_md_feat_file = "{}/processed/md_feat_231206_v2".format(
        ROOT
    )  # if you dont have features put None

    # You shouldn't change the key's names, just the values names
    dict_cols = {
        "oid": "oid",
        "time": "mjd",
        "flux": "flux_diff_ujy",
        "flux_err": "sigma_flux_diff_ujy",
        "detected": "detected",
        "band": "fid",
        "class": "alerceclass",
    }

    # The creation of windows could be improve (improve the computational efficiency)
    dict_info = {
        "type_windows": "windows",  # ['windows', 'linspace_idx', 'logspace_idx', 'logspace_times', 'linspace_logspace_times']
        "max_obs": 200,
        "list_time_to_eval": [16, 32, 64, 128, 256, 512, 1024, 2048],
        "bands_to_use": [1, 2],
        "classes_to_use": [
            "CV/Nova",
            "LPV",
            "YSO",
            "QSO",
            "RSCVn",
            "CEP",
            "EA",
            "RRLab",
            "RRLc",
            "SNIa",
            "SNII",
            "AGN",
            "EB/EW",
            "DSCT",
            "Blazar",
            "Microlensing",
            "SNIIn",
            "Periodic-Other",
            "SNIbc",
            "SLSN",
            "TDE",
            "SNIIb",
        ],
        "rm_some_ulens": {
            "rm": True,
            "path_ulens": "{}/processed/ulens_keep/ts_v9.0.1_b3000_ulensremove.csv".format(
                ROOT
            ),
        },
        "undersampling_windows": {  # Solo funciona con type_windows: windows
            "apply": False,
            "same_windows_by_folds": False,
            "keep_classes": ["SNIbc", "SNIIn", "SLSN", "TDE", "SNIIb", "Microlensing",],
        },
    }

    path_save_dataset += (
        "_" + dict_info["type_windows"] + "_" + str(dict_info["max_obs"]) + "_"
    )
    for band in dict_info["bands_to_use"]:
        path_save_dataset += str(band)

    if dict_info["undersampling_windows"]["apply"]:
        path_save_dataset += "_sampling"

    # Astronomical objects and their labels
    df_objid_label = pd.read_parquet("{}/raw/data_231206/objects.parquet".format(ROOT))
    df_objid_label = df_objid_label.reset_index()
    df_objid_label = df_objid_label[[dict_cols["oid"], dict_cols["class"], 'ra', 'dec']]

    main(
        path_lcs_file,
        path_md_feat_file,
        dict_cols,
        dict_info,
        df_objid_label,
        path_save_dataset,
        path_save_k_fold,
    )
