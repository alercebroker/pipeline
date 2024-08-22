import pandas as pd
import numpy as np
import h5py
import glob
import os

from sklearn.preprocessing import QuantileTransformer
from joblib import dump

from src.processing import expands_ids
from src.create_dataset import add_cols_h5py


def add_metadata(
    All_SNID_h5,
    all_oids,
    path_md_chunks,
    dict_cols,
    dict_info,
    path_save_dataset,
    all_partitions,
    num_folds,
):

    df_metadata = []
    for path_chunk in path_md_chunks:
        df_metadata.append(pd.read_parquet(path_chunk))
    df_metadata = pd.concat(df_metadata)

    for col in df_metadata.columns:
        if col != dict_cols["oid"]:
            df_metadata[col] = pd.to_numeric(df_metadata[col], errors="coerce")
            aux = df_metadata[col].to_numpy().copy()

            ##### Deleting nans #######
            if np.isnan(aux.mean()):
                df_metadata[col] = df_metadata[col].fillna(-9999)
            else:
                df_metadata[col] = df_metadata[col]
            df_metadata[col] = df_metadata[col].replace([np.inf, -np.inf], -9999)

    md_cols = df_metadata.columns

    if dict_info["type_windows"] == "windows":
        df_aux = all_partitions["fold_0"].copy()
        df_aux["oid_prefix"] = df_aux["oid"].str.split("_").str[0]
        df_metadata = pd.merge(
            df_metadata, df_aux, left_on="oid", right_on="oid_prefix", how="outer"
        ).dropna()
        df_metadata = df_metadata.rename(columns={"oid_y": "oid"}).drop(
            ["oid_x", "oid_prefix"], axis=1
        )[md_cols]

        # df_md_updated = []
        ##snids = df_metadata[dict_cols['oid']].values
        # for snid in all_oids:
        #    try:
        #        expands_ids(df_md_updated, snid, df_metadata, dict_snid_windows, dict_cols)
        #    except:
        #        print('oid: {} did not find in the LC files'.format(snid)) # posiblemente porque filtramos las clases
        # df_metadata = pd.concat(df_md_updated)

    df_metadata = df_metadata.set_index(dict_cols["oid"])
    df_metadata.index.names = ["SNID"]
    df_metadata = df_metadata.filter(items=All_SNID_h5.astype(str), axis=0)

    for fold in range(num_folds):
        name_used = "fold_{}".format(fold)
        aux_pd = all_partitions["fold_{}".format(fold)]
        aux_idx = aux_pd[
            aux_pd["partition"] == "training_{}".format(fold)
        ].index.to_numpy()

        qt = QuantileTransformer(
            n_quantiles=1000, random_state=0, output_distribution="uniform"
        )
        qt.fit(df_metadata.iloc[aux_idx])

        os.makedirs("{}/quantiles/metadata".format(path_save_dataset), exist_ok=True)
        dump(qt, "{}/quantiles/metadata/{}.joblib".format(path_save_dataset, name_used))

    add_cols_h5py(df_metadata, path_save_dataset, name_dataset="metadata_feat")

    return list(df_metadata.columns)


def add_features(
    All_SNID_h5,
    all_oids,
    path_md_feat_file,
    dict_cols,
    dict_info,
    path_save_dataset,
    list_times_to_eval,
    all_partitions,
    num_folds,
    df_objid_label  # used for periodic-other unbias
):

    for time_to_eval in list_times_to_eval:
        print("- adding features {} days".format(time_to_eval))
        path_feat_chunks = glob.glob(
            "{}/features/{}_days/*".format(path_md_feat_file, time_to_eval)
        )

        df_feat = []
        for path_chunk in path_feat_chunks:
            df_feat.append(pd.read_parquet(path_chunk))
        df_feat = pd.concat(df_feat)

        for col in df_feat.columns:
            if col != dict_cols["oid"]:
                df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
                aux = df_feat[col].to_numpy().copy()

                ##### Deleting nans #######
                if np.isnan(aux.mean()):
                    df_feat[col] = df_feat[col].fillna(-9999)
                else:
                    df_feat[col] = df_feat[col]
                df_feat[col] = df_feat[col].replace([np.inf, -np.inf], -9999)

        # manage bias Periodic-Other
        # we will modify the training data for fold_0,
        # so using other partitions will leak info
        assert num_folds == 1

        training_partition = "training_0"
        labels = all_partitions['fold_0']
        po_tp = labels[
            (labels["alerceclass"] == "Periodic-Other")
            & (labels["partition"] == training_partition)
        ]

        training_po_oids = list(set([oid.split('_')[0] for oid in po_tp['oid'].values]))
        del po_tp
        po_tp_coords = df_objid_label[df_objid_label["oid"].isin(training_po_oids)]
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
        to_be_replaced = list(set(training_po_oids) - set(not_to_be_replaced))

        tbr_feature_mask = df_feat["oid"].isin(to_be_replaced)
        n_replacement_needed = tbr_feature_mask.astype(int).sum()
        replacement_coords = (
            df_feat[df_feat["oid"].isin(not_to_be_replaced)][
                [f"Coordinate_{x}" for x in "xyz"]
            ]
            .sample(n_replacement_needed, replace=True)
            .values
        )
        df_feat = df_feat.set_index("oid")
        df_feat.loc[
            to_be_replaced, [f"Coordinate_{x}" for x in "xyz"]
        ] = replacement_coords
        df_feat = df_feat.reset_index()

        
        feat_cols = df_feat.columns

        if dict_info["type_windows"] == "windows":
            df_aux = all_partitions["fold_0"].copy()
            df_aux["oid_prefix"] = df_aux["oid"].str.split("_").str[0]
            df_feat = pd.merge(
                df_feat, df_aux, left_on="oid", right_on="oid_prefix", how="outer"
            ).dropna()
            df_feat = df_feat.rename(columns={"oid_y": "oid"}).drop(
                ["oid_x", "oid_prefix"], axis=1
            )[feat_cols]

            # df_feat_updated = []
            ##snids = df_feat[dict_cols['oid']].values
            # for snid in all_oids:
            #    try:
            #        expands_ids(df_feat_updated, snid, df_feat, dict_snid_windows, dict_cols)
            #    except:
            #        print('oid: {} did not find in the LC files'.format(snid)) # posiblemente porque filtramos las clases
            # df_feat = pd.concat(df_feat_updated)

        df_feat = df_feat.set_index(dict_cols["oid"])
        df_feat.index.names = ["SNID"]
        df_feat = df_feat.filter(items=All_SNID_h5.astype(str), axis=0)

        add_cols_h5py(
            df_feat,
            path_save_dataset,
            name_dataset="extracted_feat_{}".format(time_to_eval),
        )

    return list(df_feat.columns)


def add_metadata_QT_as_MLP(
    All_SNID_h5,
    all_oids,
    path_md_chunks,
    dict_cols,
    dict_info,
    path_save_dataset,
    all_partitions,
    num_folds,
):

    df_metadata = []
    for path_chunk in path_md_chunks:
        df_metadata.append(pd.read_parquet(path_chunk))
    df_metadata = pd.concat(df_metadata)

    for col in df_metadata.columns:
        if col != dict_cols["oid"]:
            df_metadata[col] = pd.to_numeric(df_metadata[col], errors="coerce")
            aux = df_metadata[col].to_numpy().copy()

            ##### Deleting nans #######
            if np.isnan(aux.mean()):
                df_metadata[col] = df_metadata[col].fillna(-9999)
            else:
                df_metadata[col] = df_metadata[col]
            df_metadata[col] = df_metadata[col].replace([np.inf, -np.inf], -9999)

    if dict_info["type_windows"] == "windows":
        pass
        # df_md_updated = []
        ##snids = df_metadata[dict_cols['oid']].values
        # for snid in all_oids:
        #    try:
        #        expands_ids(df_md_updated, snid, df_metadata, dict_snid_windows, dict_cols)
        #    except:
        #        print('oid: {} did not find in the LC files'.format(snid)) # posiblemente porque filtramos las clases
        # df_metadata = pd.concat(df_md_updated)

    df_metadata = df_metadata.set_index(dict_cols["oid"])
    df_metadata.index.names = ["SNID"]
    df_metadata = df_metadata.filter(items=All_SNID_h5.astype(str), axis=0)

    for fold in range(num_folds):
        name_used = "fold_{}".format(fold)
        aux_pd = all_partitions["fold_{}".format(fold)]
        aux_idx = aux_pd[
            aux_pd["partition"] == "training_{}".format(fold)
        ].index.to_numpy()

        qt = QuantileTransformer(
            n_quantiles=10000, random_state=0, output_distribution="normal"
        )
        qt.fit(df_metadata.iloc[aux_idx])

        os.makedirs("{}/quantiles/metadata".format(path_save_dataset), exist_ok=True)
        dump(qt, "{}/quantiles/metadata/{}.joblib".format(path_save_dataset, name_used))

    add_cols_h5py(df_metadata, path_save_dataset, name_dataset="metadata_feat")

    return list(df_metadata.columns)


def add_features_QT_as_MLP(
    All_SNID_h5,
    all_oids,
    path_md_feat_file,
    dict_cols,
    dict_info,
    dict_snid_windows,
    path_save_dataset,
    list_times_to_eval,
    all_partitions,
    num_folds,
):

    for time_to_eval in list_times_to_eval:
        print("- adding features {} days".format(time_to_eval))
        path_feat_chunks = glob.glob(
            "{}/features/{}_days/*".format(path_md_feat_file, time_to_eval)
        )

        df_feat = []
        for path_chunk in path_feat_chunks:
            df_feat.append(pd.read_parquet(path_chunk))
        df_feat = pd.concat(df_feat)

        for col in df_feat.columns:
            if col != dict_cols["oid"]:
                df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
                aux = df_feat[col].to_numpy().copy()

                ##### Deleting nans #######
                if np.isnan(aux.mean()):
                    df_feat[col] = df_feat[col].fillna(-9999)
                else:
                    df_feat[col] = df_feat[col]
                df_feat[col] = df_feat[col].replace([np.inf, -np.inf], -9999)

        if dict_info["type_windows"] == "windows":
            df_feat_updated = []
            # snids = df_feat[dict_cols['oid']].values
            for snid in all_oids:
                try:
                    expands_ids(
                        df_feat_updated, snid, df_feat, dict_snid_windows, dict_cols
                    )
                except:
                    print(
                        "oid: {} did not find in the LC files".format(snid)
                    )  # posiblemente porque filtramos las clases
            df_feat = pd.concat(df_feat_updated)

        df_feat = df_feat.set_index(dict_cols["oid"])
        df_feat.index.names = ["SNID"]
        df_feat = df_feat.filter(items=All_SNID_h5.astype(str), axis=0)

        for fold in range(num_folds):
            name_used = "fold_{}".format(fold)
            aux_pd = all_partitions["fold_{}".format(fold)]
            aux_idx = aux_pd[
                aux_pd["partition"] == "training_{}".format(fold)
            ].index.to_numpy()

            qt = QuantileTransformer(
                n_quantiles=10000, random_state=0, output_distribution="normal"
            )
            qt.fit(df_feat.iloc[aux_idx])

            os.makedirs(
                "{}/quantiles/features/{}_days".format(path_save_dataset, time_to_eval),
                exist_ok=True,
            )
            dump(
                qt,
                "{}/quantiles/features/{}_days/{}.joblib".format(
                    path_save_dataset, time_to_eval, name_used
                ),
            )

        add_cols_h5py(
            df_feat,
            path_save_dataset,
            name_dataset="extracted_feat_{}".format(time_to_eval),
        )

    return list(df_feat.columns)


def compute_feature_quantiles(dataset_path, path_save_dataset, list_time_to_eval):
    h5_file = h5py.File(dataset_path)

    all_features = []
    for time in list_time_to_eval:
        all_features.append(h5_file[f"extracted_feat_{time}"][h5_file["training_0"]])

    h5_file.close()
    all_features = np.concatenate(all_features, axis=0)
    qt = QuantileTransformer(
        n_quantiles=1000, random_state=0, output_distribution="uniform"
    )
    qt.fit(all_features)
    os.makedirs(
        f"{path_save_dataset}/quantiles/features",
        exist_ok=True,
    )
    dump(
        qt,
        f"{path_save_dataset}/quantiles/features/fold_0.joblib"
    )