import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import multiprocessing
import tqdm
import glob
import copy
import os

from utils import *


def wrapper_extract_info(args):
    path_chunk, dict_info = args
    return extract_info(path_chunk, dict_info)


def extract_info(path_chunk, dict_info):
    num_batch = path_chunk.split("_")[-1].split(".")[0]
    astro_objects_batch = pd.read_pickle("./{}".format(path_chunk))

    list_dict_feat = []
    for astro_object in astro_objects_batch:
        ao_features = astro_object["features"][
            ~astro_object["features"]["fid"].isin([None])
        ]
        ao_aux_features = astro_object["features"][
            astro_object["features"]["fid"].isin([None])
        ]

        ao_features["name_fid"] = ao_features["name"] + "_" + ao_features["fid"]

        diccionario = {}
        for _, row in ao_features.iterrows():
            diccionario[row["name_fid"]] = row["value"]
            diccionario["oid"] = astro_object["detections"].index[0]

        for _, row in ao_aux_features.iterrows():
            diccionario[row["name"]] = row["value"]

        list_dict_feat.append(diccionario)

    df_features = pd.DataFrame(list_dict_feat)

    if dict_info["i"] == 0:
        path_save_md = "{}/metadata".format(dict_info["path_save"])
        os.makedirs(path_save_md, exist_ok=True)
        df_metadata = df_features[["oid"] + dict_info["md_col_names"]]
        columns_wo_oid = [c for c in df_metadata.columns if c != 'oid']
        assert set(columns_wo_oid).issubset(set(ZTF_ff_columns_to_PROD.keys()))
        df_metadata = df_metadata.rename(
            columns=ZTF_ff_columns_to_PROD)
        df_metadata.to_parquet(
            "./{}/metadata_batch_{}.parquet".format(path_save_md, num_batch)
        )

    df_features = df_features.drop(dict_info["md_col_names"], axis="columns")
    columns_wo_oid = [c for c in df_features.columns if c != 'oid']
    assert set(columns_wo_oid).issubset(set(ZTF_ff_columns_to_PROD.keys()))
    df_features = df_features.rename(
        columns=ZTF_ff_columns_to_PROD)
    df_features.to_parquet(
        "./{}/feat_batch_{}.parquet".format(dict_info["path_save_feat_time"], num_batch)
    )


def main(dict_info, num_cores, ROOT):
    os.makedirs(dict_info["path_save"], exist_ok=True)

    for i, time_to_eval in enumerate(dict_info["list_time_to_eval"]):
        real_time = copy.copy(time_to_eval)
        if time_to_eval is None:
            real_time = 2048

        print("Time to eval {}".format(real_time))
        path_save_feat_time = "{}/features/{}_days".format(
            dict_info["path_save"], real_time
        )
        os.makedirs(path_save_feat_time, exist_ok=True)

        path_chunks = glob.glob(
            "{}/raw/data_231206_features/astro_objects_batch_{}_*".format(
                ROOT, time_to_eval
            )
        )

        dict_info.update({"path_save_feat_time": path_save_feat_time, "i": i})

        args_for_pool = [(path_chunk, dict_info) for path_chunk in path_chunks]
        with multiprocessing.Pool(processes=num_cores) as pool:
            _ = list(
                tqdm.tqdm(
                    pool.imap_unordered(wrapper_extract_info, args_for_pool),
                    total=len(path_chunks),
                )
            )


if __name__ == "__main__":
    ROOT = "./data/datasets/ZTF_ff"

    dict_info = {
        "path_save": "{}/processed/md_feat_231206_v2".format(ROOT),
        "list_time_to_eval": [16, 32, 64, 128, 256, 512, 1024, None],  #
        "md_col_names": ["W1-W2", "W2-W3", "W3-W4", "sgscore1", "dist_nr", "ps_g-r"],
    }

    num_cores = 40

    main(dict_info, num_cores, ROOT)
