import os
import re
import logging
import time
import yaml
import pandas as pd
import multiprocessing as mp
import webdataset as wds
import torch
from tqdm import tqdm

from src.data.processing.partition_manager import open_partitions, ordered_partitions
from src.data.processing.dataset_builder import create_dataset_h5py, create_dataset_h5py_bandwise

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(processName)s] %(message)s"
)


def group_lc_by_oid_and_band(df_lc, bands, fields=["time", "brightness", "e_brightness"]):
    """
    Para cada oid, agrupa las columnas de lightcurve por banda y deja columnas tipo time_g, brightness_r, etc.
    Retorna un DataFrame con una fila por oid y columnas separadas por banda.
    """
    def make_row(group):
        row = {"oid": group["oid"].iloc[0]}
        for band in bands:
            for field in fields:
                arr = group.loc[group["band"] == band, field].to_numpy()
                row[f"{field}_{band}"] = arr
        return pd.Series(row)
    df_grouped = df_lc.groupby("oid").apply(make_row).reset_index(drop=True)
    return df_grouped

def lightcurves_to_bandwise(df_lc, bands=None, fields=None):
    """
    Agrupa DataFrame de lightcurves (varias filas por oid y banda) en una fila por oid,
    con columnas por campo y banda (ej: time_g, brightness_r, ...).
    """
    if bands is None:
        bands = sorted(df_lc["band"].unique())
    if fields is None:
        # Todos los campos excepto 'oid' y 'band'
        fields = [c for c in df_lc.columns if c not in ["oid", "band"]]

    # Para cada (oid), extrae para cada banda un array/lista de cada campo
    records = []
    for oid, group in df_lc.groupby("oid"):
        row = {"oid": oid}
        for band in bands:
            band_group = group[group["band"] == band]
            for field in fields:
                colname = f"{field}_{band}"
                row[colname] = band_group[field].to_numpy()
        records.append(row)

    df_bandwise = pd.DataFrame(records).set_index("oid")
    return df_bandwise

def main(dict_info, num_cores, use_multiprocessing=True):
    logging.info("Abriendo particiones")
    path_partitions = os.path.join(
        dict_info['input_dir'],
        'partitions',
        dict_info['partition_folder'],
        'partitions.parquet'
    )
    df_partitions = pd.read_parquet(path_partitions)
    df_partitions = df_partitions.reset_index(drop=True)
    df_partitions, mapping_to_int = open_partitions(df_partitions)
    num_folds = df_partitions['partition'].str.extract(r'_(\d+)$').nunique().iloc[0]
    logging.info(f"Number of folds detected: {num_folds}")

    files = os.listdir(f"{dict_info['input_dir']}/lightcurves")
    list_chunks = [file.split('.')[0].split('_')[-1] for file in files]

    # ---- Acumular los dataframes en listas ----
    lightcurves_list = []
    metadata_list = []
    dict_features = {time_to_eval: [] for time_to_eval in dict_info["list_time_to_eval"]} if dict_info['extract_features'] else {}

    for n_chunk in list_chunks:
        path_lc = f"{dict_info['input_dir']}/lightcurves/lc_{n_chunk}.parquet"
        path_md = f"{dict_info['input_dir']}/metadata/md_{n_chunk}.parquet"

        lightcurves_list.append(pd.read_parquet(path_lc))
        if dict_info['extract_metadata']:
            metadata_list.append(pd.read_parquet(path_md))

        if dict_info['extract_features']:
            for time_to_eval in dict_info["list_time_to_eval"]:
                feat_path = f"{dict_info['input_dir']}/features/feat_{time_to_eval}days_{n_chunk}.parquet"
                dict_features[time_to_eval].append(pd.read_parquet(feat_path))

    # ---- Concatenar todos los dataframes acumulados ----
    df_lightcurves = pd.concat(lightcurves_list)
    bands = sorted(df_lightcurves["band"].unique())
    combined_info = lightcurves_to_bandwise(df_lightcurves, bands=bands)

    if dict_info['extract_metadata']:
        df_metadata = pd.concat(metadata_list)
        md_cols = list(df_metadata.columns)
        df_metadata["metadata_feat"] = list(df_metadata.to_numpy())
        df_metadata = df_metadata[["metadata_feat"]]
        df_metadata.index = df_metadata.index.astype(str)
        combined_info.index = combined_info.index.astype(str)
        combined_info = combined_info.join(df_metadata)     

    if dict_info['extract_features']:
        for time_to_eval in dict_info["list_time_to_eval"]:
            dict_features[time_to_eval] = pd.concat(dict_features[time_to_eval])
            feat_cols = list(dict_features[time_to_eval].columns)
            dict_features[time_to_eval][f"extracted_feat_{time_to_eval}days"] = list(dict_features[time_to_eval].to_numpy())
            dict_features[time_to_eval] = dict_features[time_to_eval][[f"extracted_feat_{time_to_eval}days"]]
            dict_features[time_to_eval].index = dict_features[time_to_eval].index.astype(str)
            combined_info = combined_info.join(dict_features[time_to_eval])   

    logging.info("Incorporando indices de particiones")
    all_partitions = {}
    for fold in range(num_folds):
        all_partitions["fold_%s" % fold] = ordered_partitions(
            combined_info, df_partitions, fold
        )

    combined_info = pd.merge(
        combined_info,
        all_partitions["fold_0"],
        left_index=True,
        right_on="oid",
        how="inner",
    )[["oid"] + combined_info.columns.tolist() + ["label_int", "class_name"]]
  
    logging.info("Saving dataset")
    path_save = dict_info['output_dir']
    os.makedirs(path_save, exist_ok=True)
    create_dataset_h5py_bandwise(all_partitions, combined_info, num_folds, dict_info, path_save, bands)

    dict_info.update({
        "md_cols": md_cols,
        "feat_cols": feat_cols,
        "mapping_classes": mapping_to_int,
        "bands_to_use": bands,
        })

    with open("{}/dict_info.yaml".format(path_save), "w") as f:
        yaml.dump(dict_info, f)
    logging.info("Dataset and dict_info saved successfully")

if __name__ == "__main__":
    dict_info = {
        "input_dir": "../../../data_acquisition/ztf_forced_photometry/data/processed/ds_pre250408_pos250517",
        "output_dir": "data/ztf_forced_photometry/ds_pre250408_pos250517",
        "partition_folder": "250408_ndetge8_20folds",
        "extract_metadata": True,
        "extract_features": True,
        "list_time_to_eval": [8, 16, 32, 64, 128, 256, 512, 1024, 'all'],
    }

    use_multiprocessing = False   # Cambia a False para hacerlo secuencial
    num_cores = 20

    logging.info("Starting main process")
    start_time = time.time()
    main(dict_info, num_cores, use_multiprocessing)
    end_time = time.time()
    logging.info(f"Execution completed in {end_time - start_time:.2f} seconds")
