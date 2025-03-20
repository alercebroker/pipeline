import pandas as pd
import multiprocessing
import logging
import tqdm
import glob
import time
import yaml
import os

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from src.data.processing.tabular_data_processor import processing_metadata, processing_features
from src.data.processing.lightcurve_processor import processing_lightcurve
from src.data.processing.lightcurve_utils import adapting_format_to_lc_windows
from src.data.processing.partition_manager import open_partitions, ordered_partitions
from src.data.processing.dataset_builder import create_dataset_h5py
from schema import ZTF_ff_columns_to_PROD

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

def wrapper_extract_info(args):
    path_chunk, dict_info, dict_cols, dir_astro_features = args
    return extract_info(path_chunk, dict_info, dict_cols, dir_astro_features)

def extract_info(path_chunk, dict_info, dict_cols, dir_astro_features):
    file_name = path_chunk.split("/")[-1]
    astro_objects_batch = pd.read_pickle("./{}".format(path_chunk))
    if dict_info['extract_metadata']:
        astro_objects_batch_md = pd.read_pickle('{}/None_{}'.format(dir_astro_features, file_name))
    if dict_info['extract_features']:
        dict_astro_objects_feat = dict()
        for time_to_eval in dict_info["list_time_to_eval"]:    
            dict_astro_objects_feat[time_to_eval] = pd.read_pickle('{}/{}_{}'.format(dir_astro_features, 
                                                                                     time_to_eval, 
                                                                                     file_name))
            
    filtered_cols = [col for col in dict_cols.values() if col not in ['oid', 'class_name']]
    df_dataset = []
    for i, astro_object in enumerate(astro_objects_batch):
        detections = astro_object['detections']
        forced_photometry = astro_object['forced_photometry']

        detections = detections[detections['unit'] == 'diff_flux']
        forced_photometry = forced_photometry[forced_photometry['unit'] == 'diff_flux']
        
        lightcurve = pd.concat([detections, forced_photometry])
        lightcurve = lightcurve.rename(columns=dict_cols)[filtered_cols].reset_index()
        lightcurve.sort_values(by='time', inplace=True)

        lightcurve = processing_lightcurve(lightcurve, dict_info, dict_cols)
        combined_info = lightcurve.copy()

        if dict_info['extract_metadata']:
            col_name = 'metadata_feat'
            oid = astro_objects_batch_md[i]['detections'].index[0]
            metadata = astro_objects_batch_md[i]['features']
            metadata = processing_metadata(metadata, dict_info, oid, col_name)
            combined_info = pd.merge(combined_info, metadata, on='oid', how='left')

        if dict_info['extract_features']:
            for time_to_eval, astro_objects_batch_feat in dict_astro_objects_feat.items():
                col_name = f'extracted_feat_{time_to_eval}'
                oid = astro_objects_batch_feat[i]['detections'].index[0]
                features = astro_objects_batch_feat[i]['features']
                features = processing_features(features, dict_info, oid, col_name)
                combined_info = pd.merge(combined_info, features, on='oid', how='left')

        df_dataset.append(combined_info)

    df_dataset = pd.concat(df_dataset)

    return df_dataset


def main(dict_info, dict_cols, num_cores, dir_astro_lightcurves, dir_astro_features, path_partition, use_multiprocessing=True):
    logging.info("Opening partitions")
    df_partitions = pd.read_parquet(path_partition)
    df_partitions = df_partitions.rename(columns=dict_cols).reset_index(drop=True)
    df_partitions, mapping_to_int = open_partitions(df_partitions)
    num_folds = df_partitions['partition'].str.extract(r'_(\d+)$').nunique().iloc[0]
    logging.info(f"Number of folds detected: {num_folds}")

    if dict_info['extract_metadata']: 
        dict_info['md_cols'] = [ZTF_ff_columns_to_PROD[col] for col in dict_info["md_col_names"]]

    if dict_info['extract_features']: 
        all_columns = set(ZTF_ff_columns_to_PROD.keys())
        feat_cols_names = sorted(list(all_columns - set(dict_info["md_col_names"])))
        feat_cols_names = [col for col in feat_cols_names if not any(substring in col for substring in dict_info['rm_feat'])]
        dict_info['feat_cols'] = [ZTF_ff_columns_to_PROD[col] for col in feat_cols_names]

    logging.info("Processing AstroObjects")
    path_chunks = glob.glob(f"{dir_astro_lightcurves}/astro_objects_batch_*")

    if use_multiprocessing:
        args_for_pool = [(path_chunk, dict_info, dict_cols, dir_astro_features) for path_chunk in path_chunks]
        with multiprocessing.Pool(processes=num_cores) as pool:
            df_dataset = list(
                tqdm.tqdm(
                    pool.imap_unordered(wrapper_extract_info, args_for_pool),
                    total=len(path_chunks),
                )
            )
    else:
        df_dataset = []
        for path_chunk in tqdm.tqdm(path_chunks, total=len(path_chunks)):
            df_dataset.append(wrapper_extract_info((path_chunk, dict_info, dict_cols, dir_astro_features)))

    df_dataset = pd.concat(df_dataset)
    logging.info("All chunks processed")

    if dict_info['type_windows'] == 'windows':
        logging.info("Adapting format to LC windows")
        df_dataset, df_partitions = adapting_format_to_lc_windows(df_dataset, df_partitions, num_folds)

    df_dataset = df_dataset[df_dataset.oid.isin(df_partitions.oid.unique())]
    df_dataset = df_dataset.set_index("oid")

    logging.info("Creating dataset")
    all_partitions = {}
    for fold in range(num_folds):
        all_partitions["fold_%s" % fold] = ordered_partitions(
            df_dataset, df_partitions, fold
        )

    df_dataset = pd.merge(
        df_dataset,
        all_partitions["fold_0"],
        left_index=True,
        right_on="oid",
        how="inner",
    )[["oid"] + df_dataset.columns.tolist() + ["label_int", "class_name"]]

    logging.info("Saving dataset")
    path_save = dict_info['path_save']
    os.makedirs(path_save, exist_ok=True)
    create_dataset_h5py(all_partitions, df_dataset, num_folds, dict_info, path_save)

    dict_info.update({"mapping_classes": mapping_to_int})
    del dict_info['md_col_names']

    with open("{}/dict_info.yaml".format(path_save), "w") as f:
        yaml.dump(dict_info, f)
    logging.info("Dataset and dict_info saved successfully")



if __name__ == "__main__":
    ROOT = "../../../data_acquisition/ztf_forced_photometry"

    dir_astro_lightcurves = "{}/preprocessed/data_241209_ndetge8_ao".format(ROOT)
    dir_astro_features = "{}/preprocessed/data_241209_ndetge8_ao_shorten_features".format(ROOT)
    path_partition = '{}/preprocessed/partitions/241209_ndetge8_sanchez_tax_20folds/partitions.parquet'.format(ROOT)

    timestamp = datetime.now().strftime("%y%m%d")
    dict_info = {
        "path_save": "data/ztf_forced_photometry/processed/ds_pre241209_pos{}_detff_ndetge8_sanchez_tax_20folds".format(timestamp),
        "type_windows": "windows", 
        "max_obs": 200,
        "extract_metadata": True,
        "extract_features": True,
        "list_time_to_eval": [8, 16, 32, 64, 128, 256, 512, 1024, None],
        "md_col_names": ["W1-W2", "W2-W3", "W3-W4", "sgscore1", "distpsnr1", "ps_g-r", 'ps_r-i', 'ps_i-z'],
        "rm_feat": ['TDE_mag0', 'fleet_t0', 'fleet_m0', 'ulens_t0', 'ulens_mag0'],
        "bands_to_use": ['g', 'r'] # NO esta haciendo nada aqui, pero se usa en el entrenamiento, quizas agregarlo al dict_info desde los mismos AO
    }

    # The keys refer to the names of the dataset 
    # The values refer to the news names to use (keep fixed)
    dict_cols = { 
        "oid": "oid",
        "mjd": "time",
        "brightness": "brightness",
        "e_brightness": "e_brightness",
        "detected": "detected",
        "fid": "band",
        "alerceclass": "class_name",
    }

    use_multiprocessing = True
    num_cores = 20

    logging.info("Starting main process")
    start_time = time.time()

    main(dict_info, 
         dict_cols, 
         num_cores, 
         dir_astro_lightcurves, 
         dir_astro_features, 
         path_partition,
         use_multiprocessing)

    end_time = time.time()

    execution_time = end_time - start_time
    logging.info(f"Execution completed in {execution_time:.2f} seconds")